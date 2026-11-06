# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The greedy TAIL set beside the sampler set of a device-sampler single-slot chain (fakes, no device)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_epilogue_warm_capture_no_device import (
    FakeTensor,
    extension_with_fakes,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm


def test_greedy_epilogue_stops_after_the_resolve(monkeypatch):
    ext, keys = extension_with_fakes(monkeypatch)
    row_io = FakeTensor("resident token row")
    trace_output = SimpleNamespace(logits=FakeTensor("logits"))
    candidates, token_row = ext.capture_epilogue(trace_output, row_io, greedy_only=True)
    assert ext.lm_head.resolves == ["into=resident token row"]  # the same resolve form as the sampler set's
    assert keys == [] and step_copies() == []  # no sampler call, no token-row copy
    assert token_row.role == "resolved token row" and ext.trace_rows == [] and ext.trace_logits == []
    assert ext.greedy_trace_logits == [trace_output.logits]
    ext.capture_epilogue(trace_output, row_io)  # the sampler set
    assert len(keys) == 1 and ext.trace_rows[-1].role == "candidate row" and len(ext.trace_logits) == 1


def step_copies():
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step

    return step.ttnn.copies


def test_full_logits_come_from_the_set_that_ran(monkeypatch):
    ext, _ = extension_with_fakes(monkeypatch)
    seen = []
    monkeypatch.setattr(ext, "_gather", lambda logits: seen.append(logits))
    ext.trace_logits, ext.greedy_trace_logits = ["s0", "s1"], []
    ext.read_full_logits(1)
    ext.read_full_logits(1, sampled=True)
    ext.greedy_trace_logits = ["g0", "g1"]
    ext.read_full_logits(1)
    ext.read_full_logits(0, sampled=True)
    assert seen == ["s1", "s1", "g1", "s0"]


def make_chain(monkeypatch, *, two_sets: bool):
    calls: list[int] = []
    monkeypatch.setattr(session, "ttnn", SimpleNamespace(_ttnn_execute_trace=lambda mesh, tid, **kw: calls.append(tid)))
    chain = object.__new__(session.Qwen38TracedChain)
    chain.mtp = None
    chain.construction = SimpleNamespace(builder=SimpleNamespace(mesh_device="mesh"))
    chain.head_trace_ids, chain.tail_trace_ids = [10, 11, 12, 13], [20, 21, 22, 23]
    chain.greedy_head_trace_ids = [30, 31, 32, 33] if two_sets else []
    chain.greedy_tail_trace_ids = [40, 41, 42, 43] if two_sets else []
    chain.prove_ple_reads = lambda: None
    return chain, calls


def test_execute_routes_by_the_flag(monkeypatch):
    chain, calls = make_chain(monkeypatch, two_sets=True)
    chain.execute_head(2)
    chain.execute_tail(2)
    chain.execute_head(2, sampled=True)
    chain.execute_tail(3, sampled=True)
    assert calls == [32, 42, 12, 23]  # the default is the greedy set


def test_one_set_ignores_the_flag(monkeypatch):
    chain, calls = make_chain(monkeypatch, two_sets=False)
    chain.execute_head(1)
    chain.execute_tail(1)
    chain.execute_head(1, sampled=True)
    chain.execute_tail(1, sampled=True)
    assert calls == [11, 21, 11, 21]


def test_trace_ids_list_both_sets(monkeypatch):
    chain, _ = make_chain(monkeypatch, two_sets=True)
    chain.chunk_trace_id = chain.long_chunk_trace_id = chain.slab_trace_id = None
    chain.mtp_chains = {}
    assert sorted(chain.trace_ids()) == sorted([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43])


def test_open_refuses_a_second_set_without_the_sampler(expect_error):
    with expect_error(ValueError, "greedy_tail_set needs device_sampler"):
        session.Qwen38TracedChain.open(None, marker=print, sampling=True, greedy_tail_set=True)
    with expect_error(ValueError, "greedy_tail_set must be a bool"):
        session.Qwen38TracedChain.open(None, marker=print, greedy_tail_set=1)


def test_capture_rounds_follow_the_switch():
    source = inspect.getsource(session.Qwen38TracedChain.open)
    assert "for round_is_greedy in ((False, True) if greedy_tail_set else (False,))" in source
    assert "(chain.greedy_head_trace_ids if greedy_round else chain.head_trace_ids).append(" in source
    assert "(chain.greedy_tail_trace_ids if greedy_round else chain.tail_trace_ids).append(" in source
    assert "greedy_only=greedy_round" in source
    # the greedy round is the second round: the sampler set's captures run first, as with one set


def test_adapter_opens_the_second_set_only_with_the_device_sampler():
    source = inspect.getsource(vllm.Qwen38ForCausalLM.initialize_vllm_model)
    assert 'options["greedy_tail_set"] = True' in source
    block = source[source.index('options["device_sampler"] = True') : source.index("elif os.environ.get")]
    assert 'options.get("mtp") is None' in block


def test_trace_set_kwargs_keep_the_default_call():
    assert vllm.trace_set_kwargs(False) == {} and vllm.trace_set_kwargs(True) == {"sampled": True}


class FakeChain:
    def __init__(self):
        self.calls: list[tuple] = []

    def feed_token_row(self, token):
        return True

    write_token_row = feed_token_row

    def prepare_sampled_step(self):
        pass

    def execute_head(self, residue, **kwargs):
        self.calls.append(("head", residue, kwargs))

    def execute_tail(self, residue, **kwargs):
        self.calls.append(("tail", residue, kwargs))

    def refresh_ple_row(self, token, context):
        return context


@pytest.mark.parametrize("sampled, expected", [(False, {}), (True, {"sampled": True})])
def test_adapter_step_passes_the_set_through(monkeypatch, sampled, expected):
    monkeypatch.delenv(vllm.PLE_EARLY_DECODE_VARIABLE, raising=False)
    adapter = object.__new__(vllm.Qwen38ForCausalLM)
    adapter.chain = FakeChain()
    adapter.slot = SimpleNamespace(committed=[1, 2, 3, 4, 5], ple_context=None)
    residue = adapter._step(7, sampled=sampled) if sampled else adapter._step(7)
    assert residue == 1 and adapter.slot.committed[-1] == 7
    assert adapter.chain.calls == [("head", 1, expected), ("tail", 1, expected)]


# -- the standard sampler (ttsampler) on the sampler set ---------------------------------------------------------


class FakeTTSampler:
    def __init__(self, events):
        self.events = events
        self.generator = SimpleNamespace(
            tt_sampling=SimpleNamespace(
                k_tensor="k",
                p_tensor="p",
                temp_tensor="t",
                _greedy_col="g",
                seeds_tt_tensor="s",
                user_ids_tt_tensor="u",
            )
        )

    def sample(self, logits, out_tok):
        self.events.append(("sample", logits, out_tok))

    def token_row_from_out(self, out_tok):
        self.events.append(("token_row_from_out", out_tok))
        return FakeTensor("sampled row")

    def write_step_seeds(self):
        self.events.append(("seeds",))


def ttsampler_extension(monkeypatch):
    ext, keys = extension_with_fakes(monkeypatch)
    events: list = []
    step = step_module()
    step.ttnn.clone = lambda tensor: events.append(("clone", tensor)) or "owned logits"
    step.ttnn.pad = lambda tensor, padding, value: events.append(("pad", tensor, padding, value)) or "padded"
    step.ttnn.copy = lambda source, target: events.append(("copy", source.role, target.role))
    step.ttnn.deallocate = lambda tensor: events.append(("deallocate", tensor))
    ext.enable_ttsampler(FakeTTSampler(events), "out_tok")
    logits = SimpleNamespace(tensor="logit tensor", global_shape=(1, 1, 1, 248320))
    return ext, keys, events, logits


def step_module():
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step

    return step


def test_ttsampler_sampler_set_pads_samples_and_copies_in_order(monkeypatch):
    ext, keys, events, logits = ttsampler_extension(monkeypatch)
    row_io = FakeTensor("resident token row")
    candidates, token_row = ext.capture_epilogue(SimpleNamespace(logits=logits), row_io)
    kinds = [event[0] for event in events]
    assert kinds == ["clone", "pad", "sample", "deallocate", "deallocate", "token_row_from_out", "copy", "deallocate"]
    assert events[0] == ("clone", "logit tensor")
    assert events[1][1:] == ("owned logits", [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
    assert events[2][1:] == ("padded", "out_tok") and events[6] == ("copy", "sampled row", "resident token row")
    assert keys == [] and ext.lm_head.resolves == []  # neither the model's sampler nor the greedy resolve ran
    assert candidates is None and token_row is None
    assert ext.trace_rows == [None] and ext.trace_logits == [logits] and ext.greedy_trace_logits == []
    ext.read_full_logits(0, sampled=True)  # the host fallback still gathers the sampler set's logits


def test_ttsampler_greedy_set_is_unchanged(monkeypatch):
    ext, keys, events, logits = ttsampler_extension(monkeypatch)
    row_io = FakeTensor("resident token row")
    candidates, token_row = ext.capture_epilogue(SimpleNamespace(logits=logits), row_io, greedy_only=True)
    assert events == [] and ext.lm_head.resolves == ["into=resident token row"] and keys == []
    assert token_row.role == "resolved token row" and ext.greedy_trace_logits == [logits]


def test_ttsampler_warm_runs_the_variant_with_seeds_first(monkeypatch):
    ext, keys, events, logits = ttsampler_extension(monkeypatch)
    ext.warm(logits=logits, token_row_io=FakeTensor("resident token row"), label="warm")
    kinds = [event[0] for event in events]
    start = kinds.index("seeds")  # the sampler variant runs last, after the greedy passes
    assert kinds[start : start + 6] == ["seeds", "clone", "pad", "sample", "deallocate", "deallocate"]
    assert [e for e in events[:start] if e[0] in ("clone", "pad", "sample")] == []


def test_ttsampler_epilogue_never_frees_the_input_logits(monkeypatch):
    ext, keys, events, logits = ttsampler_extension(monkeypatch)
    ext.capture_epilogue(SimpleNamespace(logits=logits), FakeTensor("resident token row"))
    freed = [event[1] for event in events if event[0] == "deallocate"]
    assert "logit tensor" not in freed and freed[:2] == ["padded", "owned logits"]
    events.clear()
    ext.warm(logits=logits, token_row_io=FakeTensor("resident token row"), label="warm")
    assert "logit tensor" not in [event[1] for event in events if event[0] == "deallocate"]


def test_ttsampler_host_written_tensors(monkeypatch):
    ext, _, _, _ = ttsampler_extension(monkeypatch)
    assert ext.ttsampler_host_written() == ["k", "p", "t", "g", "s", "u"]


def test_prepare_sampled_step_writes_the_seeds(monkeypatch):
    chain, _ = make_chain(monkeypatch, two_sets=True)
    chain.ttsampler = None
    chain.prepare_sampled_step()
    events: list = []
    chain.ttsampler = FakeTTSampler(events)
    chain.prepare_sampled_step()
    assert events == [("seeds",)]


def test_open_validates_ttsampler(expect_error):
    with expect_error(ValueError, "ttsampler must be a bool"):
        session.Qwen38TracedChain.open(None, marker=print, ttsampler=1)
    with expect_error(ValueError, "ttsampler needs sampling and greedy_tail_set and no mtp"):
        session.Qwen38TracedChain.open(None, marker=print, ttsampler=True, sampling=True, device_sampler=True)
    with expect_error(ValueError, "greedy_tail_set needs device_sampler and no mtp"):  # mtp is refused first
        session.Qwen38TracedChain.open(
            None, marker=print, ttsampler=True, sampling=True, device_sampler=True, greedy_tail_set=True, mtp=1
        )
