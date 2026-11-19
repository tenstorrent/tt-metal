# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the independent batched HF probe; no TTNN imports."""
from pathlib import Path

import torch

from . import test_decode_validation_cpu as decode_fixture
from . import test_sampling_params_cpu as sampling_fixture
from .probe_full_model_batch_hf import (
    accuracy,
    greedy_params_for_order,
    load_reference,
    prediction_record,
    state_field_differences,
)


def test_exact_reference_history_not_replaced_by_scored_top1():
    path = Path(__file__).resolve().parents[1] / "readiness_aime24_chat.refpt"
    prompt, generated, topk, prompts = load_reference(path, 32)
    assert [len(p) for p in prompts] == list(range(203, 235))
    assert generated[53] != topk[53, 0]
    for request, prefix in enumerate(prompts):
        assert torch.equal(prefix, torch.cat((prompt, generated[:request])))


def test_prediction_metric_direction_and_gate():
    candidates = torch.arange(100)
    assert prediction_record(4, candidates, 0, 0, "decode")["top5"]
    assert not prediction_record(5, candidates, 0, 0, "decode")["top5"]
    assert not prediction_record(100, candidates, 0, 0, "decode")["top100"]
    records = [prediction_record(0, candidates, 0, i, "decode") for i in range(100)]
    records[:2] = [prediction_record(5, candidates, 0, i, "decode") for i in range(2)]
    assert accuracy(records)["passed"]
    records[2] = prediction_record(5, candidates, 0, 2, "decode")
    assert not accuracy(records)["passed"]


def test_ragged_causal_schedule_and_migration():
    baseline = {}
    for repeat in (0, 1):
        order = list(range(32))
        observed = {request: [request] for request in order}
        for step in range(99):
            if repeat and step == 32:
                order = list(reversed(order))
            for request in order:
                if request + step + 1 < 100:
                    target = request + step + 1
                    forced_index = request + step
                    assert forced_index == target - 1
                    observed[request].append(target)
        for request, targets in observed.items():
            assert targets == list(range(request, 100))
            assert len(targets) - 1 >= 68
        if repeat:
            assert observed == baseline
        else:
            baseline = observed


def test_actual_setter_seed_scalar_is_not_broadcast_but_probe_lists_are():
    fixture_type = sampling_fixture.SamplingParamsTest
    fixture_type.setUpClass()
    fixture = fixture_type()
    fixture.setUp()
    fixture.gen.max_batch_size = 32
    slots = tuple(range(32))
    scalar = fixture.params(temperature=0.0, top_k=1, top_p=1.0, seed=0)
    fixture.setter(fixture.gen, scalar, slots=slots)
    assert fixture.gen.request_seeds == [0] + [None] * 31
    explicit = greedy_params_for_order(fixture.params, [0] * 32, slots)
    fixture.setter(fixture.gen, explicit, slots=slots)
    assert fixture.gen.request_seeds == [0] * 32
    assert fixture.gen.formatted_sampling_params.temperature == [1.0] * 32
    assert fixture.gen.formatted_sampling_params.top_k == [1] * 32
    assert fixture.gen.formatted_sampling_params.top_p == [0.0] * 32
    # Migration reload omits the slots argument in actual decode_forward.
    reordered = greedy_params_for_order(fixture.params, list(range(100, 132)), tuple(reversed(slots)))
    fixture.setter(fixture.gen, reordered)
    assert fixture.gen.request_seeds == list(range(131, 99, -1))


def test_actual_decode_migration_resets_request_seeds_at_next_absolute_draw():
    fixture_type = decode_fixture.DecodeValidationTest
    fixture_type.setUpClass()
    fixture = fixture_type()
    fixture.setUp()
    params = greedy_params_for_order(fixture.params, [101, 202], (1, 0))
    fixture.g.decode_forward(
        torch.tensor([1, 2]),
        [10, 9],
        page_table=None,
        kv_cache=None,
        sample_on_device=True,
        read_from_device=False,
        slot_remap=(1, 0),
        reload_sampling_params=True,
        reset_sampling_state=True,
        sampling_params=params,
    )
    resets = [event for event in fixture.events if isinstance(event, tuple) and event[0] == "seed_reset"]
    assert resets == [("seed_reset", [202, 101], [11, 10], [0, 1])]
    assert fixture.g.request_seeds == [202, 101] + [None] * 30


def test_field_hash_mismatch_reports_seed_and_missing_state_without_hiding_other_fields():
    baseline = {"0": {"seed.0": "aaa", "0.layer.recurrent.0": "bbb"}, "1": {"feedback.0": "ccc"}}
    actual = {"0": {"seed.0": "ddd", "0.layer.recurrent.0": "bbb"}, "1": {}}
    assert state_field_differences(baseline, actual) == {
        "0": {"seed.0": {"expected": "aaa", "actual": "ddd"}},
        "1": {"feedback.0": {"expected": "ccc", "actual": None}},
    }
    assert state_field_differences(baseline, baseline) == {}
