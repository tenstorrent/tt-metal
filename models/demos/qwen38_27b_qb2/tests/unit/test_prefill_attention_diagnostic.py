# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from models.demos.qwen38_27b_qb2.tests.prefill_attention_diagnostic import (
    CASES,
    VARIANTS,
    configuration,
    selected_reference,
    validate_report,
)


def receipt():
    cases = []
    for batch, start, length in CASES:
        arms = []
        for name in VARIANTS:
            native = name in ("native", "native_repeat")
            checks = [
                dict(
                    passed=not native,
                    pcc_per_user=[0.9995] * batch,
                    relative_rms_per_user=[0.024 if native else 0.005] * batch,
                )
            ] * 4
            arms.append(
                dict(
                    name=name,
                    configuration=configuration(name),
                    output_sha256_per_rank=["0" * 64] * 4,
                    cache_unchanged=True,
                    query_unchanged=True,
                    cache_sha256_per_rank=[["1" * 64] * 4] * 2,
                    accuracy_per_rank=copy.deepcopy(checks),
                )
            )
        cases.append(
            dict(
                batch=batch,
                start_pos=start,
                chunk_tokens=length,
                independent_reference_checked=True,
                expected_cache_sha256=["1" * 64] * 2,
                arms=arms,
            )
        )
    return dict(state="completed", cleanup_completed=True, device_ids=[0, 4, 12, 8], cases=cases)


def test_a_reproduced_failure_is_diagnostic_evidence_not_model_qualification():
    result = validate_report(receipt())
    assert result["diagnostic_completed"] is True
    assert result["model_qualified"] is False and result["performance_gain_measured"] is False
    assert all("native" not in r["passing_variants"] for r in result["findings"])


@pytest.mark.parametrize("defect", ["variants", "inputs", "reference", "native_repeat", "threshold", "cache"])
def test_incomplete_or_weakened_diagnostic_evidence_rejected(defect, expect_error):
    report = receipt()
    row = report["cases"][0]
    if defect == "variants":
        row["arms"].pop()
    elif defect == "inputs":
        row["arms"][0]["configuration"]["exp_approx_mode"] = False
    elif defect == "reference":
        row["independent_reference_checked"] = False
    elif defect == "native_repeat":
        row["arms"][-1]["output_sha256_per_rank"] = ["2" * 64] * 4
    elif defect == "threshold":
        for arm in (row["arms"][0], row["arms"][-1]):
            arm["accuracy_per_rank"][0]["passed"] = True
    else:
        row["arms"][0]["cache_sha256_per_rank"] = [["2" * 64] * 4] * 2
    with expect_error(ValueError, ".+"):
        validate_report(report)


def test_configuration_changes_one_numerical_axis_at_a_time():
    native = configuration("native")
    assert configuration("native_repeat") == native
    assert configuration("accurate_exp") == dict(native, exp_approx_mode=False)
    accum = configuration("fp32_accum")
    assert accum["exp_approx_mode"] is None
    assert configuration("accurate_exp_fp32") == dict(accum, exp_approx_mode=False)
    high = configuration("hifi4_accurate_fp32")
    assert high == dict(
        configuration("accurate_exp_fp32"), compute_kernel=dict(accum["compute_kernel"], math_fidelity="HiFi4")
    )


def test_selected_reference_masks_future_rows_and_preserves_shuffled_prefixes():
    rng = torch.Generator().manual_seed(9102)
    query = torch.randn(2, 6, 65, 8, generator=rng).bfloat16()
    table = torch.randperm(10, generator=rng).reshape(2, 5)
    caches = [torch.randn(10, 1, 32, 8, generator=rng).bfloat16() for _ in range(2)]
    expected, independent = selected_reference(query, caches, table, 32, [0, 32, 64])
    assert torch.allclose(expected, independent, atol=2e-6, rtol=2e-5)
    modified = [c.clone() for c in caches]
    for user in range(2):
        for position in range(97, 160):
            for cache in modified:
                cache[table[user, position // 32], 0, position % 32] = 100
    changed, _ = selected_reference(query, modified, table, 32, [0, 32, 64])
    assert torch.equal(expected, changed)
