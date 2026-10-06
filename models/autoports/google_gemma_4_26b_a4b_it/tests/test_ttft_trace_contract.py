# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for prefill capture cleanup and sampling invalidation."""

from unittest.mock import Mock, call

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.common.sampling.generator import SamplingParams


def generator_stub():
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.mesh = object()
    generator.host_sampling = False
    generator.sampled_mode = False
    generator.prefill_prepared = {"cache": object()}
    generator.prefill_trace_id = None
    generator.prefill_sampling_trace_id = None
    generator.trace_id = 41
    generator.output_trace_id = None
    generator._trace_returns_logits = False
    generator.sampler = Mock()
    generator._reset_sampling_seeds = Mock()
    generator.counters = {}
    return generator


@pytest.mark.parametrize("failure", ["first_begin", "model", "second_begin", "sampling", "second_end"])
def test_failed_prefill_capture_closes_started_traces_and_releases_ids(monkeypatch, failure, expect_error):
    generator = generator_stub()
    events = []
    started = []
    ended = []
    error = RuntimeError("injected capture failure")

    def begin(mesh, *, cq_id):
        index = len(started)
        if failure == ("first_begin" if index == 0 else "second_begin"):
            raise error
        trace = 101 + index
        started.append(trace)
        events.append(("begin", trace))
        return trace

    def end(mesh, trace, *, cq_id):
        ended.append(trace)
        events.append(("end", trace))
        if trace == 102 and failure == "second_end":
            raise error

    def release(mesh, trace):
        events.append(("release", trace))

    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", end)
    monkeypatch.setattr(ttnn, "release_trace", release)
    generator._prefill_trace_step = Mock(side_effect=error if failure == "model" else None)
    generator._prefill_sampling_step = Mock(side_effect=error if failure == "sampling" else None)
    with expect_error(RuntimeError, "injected capture failure") as caught:
        generator._capture_prefill()
    assert caught.value is error
    assert ended == started
    assert [trace for operation, trace in events if operation == "release"] == started + [41]
    for trace in started:
        assert events.index(("end", trace)) < events.index(("release", trace))
    assert generator.prefill_trace_id is None
    assert generator.prefill_sampling_trace_id is None
    assert generator.trace_id is None
    assert generator.counters.get("prefill_captures", 0) == 0
    generator.sampler.reset_trace.assert_called_once_with()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(temperature=0.7, top_k=16),
        dict(presence_penalty=0.5),
        dict(frequency_penalty=0.3),
        dict(repetition_penalty=1.1),
    ],
)
def test_ineligible_sampling_invalidates_prepared_state_despite_reuse(monkeypatch, overrides):
    generator = generator_stub()
    generator.prefill_trace_id = 101
    generator.prefill_sampling_trace_id = 102
    released = Mock()
    monkeypatch.setattr(ttnn, "release_trace", released)
    values = dict(temperature=0.0, top_k=1, top_p=1.0, seed=71)
    values.update(overrides)
    generator.configure_sampling(SamplingParams(**values), _reuse_trace=True)
    assert generator.prefill_prepared is None
    assert generator.prefill_trace_id is None
    assert generator.prefill_sampling_trace_id is None
    assert generator.trace_id is None
    assert released.call_args_list == [call(generator.mesh, trace) for trace in (101, 102, 41)]
    generator.sampler.reset_trace.assert_called_once_with()
    generator._reset_sampling_seeds.assert_called_once()
    generator.sampler.reset_sampling_params.assert_called_once()


def test_neutral_padded_greedy_parameters_keep_prepared_state(monkeypatch):
    generator = generator_stub()
    prepared = generator.prefill_prepared
    generator.prefill_trace_id = 101
    generator.prefill_sampling_trace_id = 102
    release = Mock()
    monkeypatch.setattr(ttnn, "release_trace", release)
    generator.configure_sampling(
        SamplingParams(temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32), _reuse_trace=True
    )
    assert generator.prefill_prepared is prepared
    assert (generator.prefill_trace_id, generator.prefill_sampling_trace_id, generator.trace_id) == (101, 102, 41)
    assert not generator.sampled_mode
    release.assert_not_called()
    generator.sampler.reset_trace.assert_not_called()


@pytest.mark.parametrize("replace_table", [False, True])
def test_prefill_page_refreshes_only_changed_tables_and_owns_snapshot(monkeypatch, replace_table):
    generator = generator_stub()
    generator.sampler._penalties_active = False
    generator.sampler._log_probs_active = False
    generator.prefill_trace_id = 101
    generator.prefill_sampling_trace_id = 102
    key = (32, "same-cache-and-shape")
    generator._serving_prefill_key = Mock(return_value=key)
    tables = [torch.arange(8, dtype=torch.int32)[None], torch.arange(8, 16, dtype=torch.int32)[None]]
    devices = (object(), object())
    state = dict(
        key=key,
        cache=object(),
        tokens=object(),
        tables=devices,
        table_host=generator._clone_tables(tables),
        output=object(),
    )
    generator.prefill_prepared = state
    copies = []
    generator._copy = lambda value, target, counter: copies.append((value.clone(), target, counter))
    execute = Mock()
    monkeypatch.setattr(ttnn, "execute_trace", execute)

    def replay():
        copies.clear()
        output = generator.serving_prefill_tokens(
            torch.full((1, 32), 100), page_table=tables, kv_cache=state["cache"], prompt_lens=[32]
        )
        assert output is state["output"]
        assert sum(counter == "prefill_token_refreshes" for _, _, counter in copies) == 1
        return [(value, target) for value, target, counter in copies if counter == "prefill_page_refreshes"]

    snapshot = state["table_host"]
    assert replay() == []
    tables = [table.clone() for table in tables]
    assert replay() == []  # Equal contents in new host objects need no device copy.
    assert state["table_host"] is snapshot
    if replace_table:
        tables[1] = tables[1].clone()
    tables[1][0, 0] = 17
    changes = replay()
    assert len(changes) == 1 and changes[0][1] is devices[1]
    assert torch.equal(changes[0][0], tables[1])
    assert int(snapshot[1][0, 0]) == 8
    for saved, incoming in zip(state["table_host"], tables):
        assert saved.data_ptr() != incoming.data_ptr()
        assert torch.equal(saved, incoming)
    snapshot = state["table_host"]
    tables[1][0, 1] = 18  # Mutation must not silently update the cached snapshot.
    assert int(snapshot[1][0, 1]) == 9
    changes = replay()
    assert len(changes) == 1 and changes[0][1] is devices[1]
    assert torch.equal(changes[0][0], tables[1])
    assert state["table_host"][1].data_ptr() != tables[1].data_ptr()
    assert replay() == []
    assert execute.call_args_list == [
        call(generator.mesh, trace, cq_id=0, blocking=False) for _ in range(5) for trace in (101, 102)
    ]


def test_warmed_prefill_only_request_captures_without_decode(monkeypatch):
    generator = generator_stub()
    generator.trace_id = None
    generator.sampler._penalties_active = False
    generator.sampler._log_probs_active = False
    key = (32, "same-cache")
    generator._serving_prefill_key = Mock(return_value=key)
    table = torch.arange(8, dtype=torch.int32)[None]
    state = dict(
        key=key,
        cache=object(),
        tokens=object(),
        tables=object(),
        table_host=table.clone(),
        output=object(),
        warmed=True,
    )
    generator.prefill_prepared = state
    generator._copy = Mock()
    events = []

    def capture():
        events.append("capture")
        generator.prefill_trace_id = 101
        generator.prefill_sampling_trace_id = 102

    generator._capture_prefill = capture
    monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, trace, **kwargs: events.append(trace))
    output = generator.serving_prefill_tokens(
        torch.ones(1, 32, dtype=torch.long),
        page_table=table,
        kv_cache=state["cache"],
        prompt_lens=[32],
    )
    assert output is state["output"]
    assert events == ["capture", 101, 102]
    assert generator.counters["prefill_replays"] == 1
