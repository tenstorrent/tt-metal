# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or mislabeled padding-isolation hardware evidence."""

import copy

import pytest

from models.demos.qwen38_27b_qb2.tests.compact_gdn import changing_input_checkpoints
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_padding import CASES, LAYERS, validate_report


def receipt():
    a, b = ([str(r + offset) * 64 for r in range(4)] for offset in (0, 4))
    checks = [dict(allocation=i, norm_sha256=h, output_sha256=h) for i, h in ((0, a), (1, b), (0, a))]
    timings = [dict(variant=v, traced_call_us=[t] * 5) for v, t in (("native", 100), ("fused", 80), ("native", 101))]
    cases = [
        dict(
            batch=batch,
            placement=p,
            mode=list(m),
            passed=True,
            padding_experiment=True,
            input_and_address_stability=True,
            changed_input_trace=True,
            output_padding_zero=True,
            input_cb_slots=2,
            max_items_per_core=3,
            checks=copy.deepcopy(checks),
            poison_checks=copy.deepcopy(checks),
            replay_checks=[dict(input_padding=m, output_sha256=[a, b, a]) for m in ("skip", "poison")],
            timing_input_padding=["zero", "skip", "zero"],
            timings=copy.deepcopy(timings),
            comparison=compare_timings(timings),
        )
        for batch, p, m in CASES
    ]
    layers = [
        dict(
            batch=b,
            input_padding=p,
            updates=4096,
            all_ranks_bit_identical=True,
            persistent_sessions_precede_trace_capture=True,
            checkpoints=[
                dict(step=s, all_values_finite=True, recurrent_conv_projected_sha256=[a, a, a])
                for s in changing_input_checkpoints(4096)
            ],
        )
        for b, p in LAYERS
    ]
    return dict(state="completed", cleanup_completed=True, device_ids=[0, 4, 12, 8], cases=cases, layers=layers)


def test_component_and_layer_receipt_never_qualifies_full_model():
    result = validate_report(receipt())
    assert result["correctness_passed"] is True
    assert result["full_model_qualified"] is False
    assert result["promoted_to_serving"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(cleanup_completed=False),
        lambda r: r["cases"].pop(),
        lambda r: r["cases"][0].update(output_padding_zero=False),
        lambda r: r["cases"][0]["poison_checks"][0].update(output_sha256=["0" * 64] * 4),
        lambda r: r["cases"][0]["checks"][0]["norm_sha256"].pop(),
        lambda r: r["cases"][0]["replay_checks"].pop(),
        lambda r: r["cases"][0].update(timing_input_padding=["zero", "poison", "zero"]),
        lambda r: r["cases"][0]["comparison"].update(qualified_speedup=10),
        lambda r: r["cases"][6].update(max_items_per_core=2),
        lambda r: r["layers"].pop(),
        lambda r: r["layers"][0]["checkpoints"].pop(),
        lambda r: r["layers"][0].update(updates=64),
        lambda r: r["layers"][0]["checkpoints"][0].update(all_values_finite=False),
    ],
)
def test_incomplete_correctness_or_timing_is_rejected(mutate, expect_error):
    report = receipt()
    mutate(report)
    with expect_error(ValueError, ".+"):
        validate_report(report)


def test_unstable_timing_does_not_get_a_gain():
    report = receipt()
    case = report["cases"][0]
    case["timings"][2]["traced_call_us"] = [150] * 5
    case["comparison"] = compare_timings(case["timings"])
    assert validate_report(report)["timings"][0]["qualified_speedup"] is None
