# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from models.demos.qwen38_27b_qb2.tests.attention_placement import LONG_CONTEXT_LAYOUTS, placement
from models.demos.qwen38_27b_qb2.tests.attention_placement_long import CASES, VARIANTS, comparison
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry


def fixture():
    cases = []
    for name, chunk in VARIANTS:
        case = geometry(262016, 8)
        case.update(
            seed=123,
            page_table_sha256="pages",
            query_bf16_sha256="query",
            reference_fp32_sha256="reference",
            precision_mode="accurate",
            worker_grid=[12, 10],
            placement=placement(name, 8, (12, 10)),
            passed=True,
            selection=dict(timing_comparison_qualified=True),
            candidates=[
                dict(
                    chunk=chunk,
                    accuracy_passed=True,
                    median_traced_call_us=1000,
                    output_fp32_sha256_per_rank=["same"] * 4,
                )
            ],
            baseline_repeat_output_fp32_sha256_per_rank=["same"] * 4,
        )
        cases.append(case)
    return cases


@pytest.mark.parametrize("batch", [4, 8, 16, 32])
def test_new_locations_preserve_work_and_fit_the_actual_grid(batch):
    for name in LONG_CONTEXT_LAYOUTS:
        p = placement(name, batch, (12, 10))
        points = [tuple(c) for c in p["logical_cores"]]
        assert len(set(points)) == len(points) == p["grid"][0] * p["grid"][1]
        assert all(0 <= x < 12 and 0 <= y < 10 for x, y in points)
        assert p["active_cores"] == p["cores_per_user"] * batch <= len(points)
        assert points == sorted(points, key=lambda c: (c[1], c[0]))
    row, outer = (placement(n, batch, (12, 10)) for n in ("row_major_96", "outer_columns_96"))
    assert row["active_cores"] == outer["active_cores"]
    assert row["logical_cores"] != outer["logical_cores"]
    assert sum(x < 6 for x, _ in outer["logical_cores"]) == 48
    assert (
        placement("full_grid_sharded", batch, (12, 10))["cores_per_user"]
        == placement("native", batch, (12, 10))["cores_per_user"]
    )


def test_first_cases_focus_on_highest_long_context_concurrency():
    assert CASES[:2] == ((262016, 8), (131072, 16))
    for length, batch in CASES:
        assert geometry(length, batch)["aligned_capacity"] % 512 == 0


def test_report_rejects_fast_inaccurate_or_nondeterministic_output():
    cases = fixture()
    cases[5]["passed"] = cases[5]["candidates"][0]["accuracy_passed"] = False
    cases[5]["selection"] = None
    cases[5]["candidates"][0]["median_traced_call_us"] = 1
    cases[9]["baseline_repeat_output_fp32_sha256_per_rank"][0] = "changed"
    cases[9]["candidates"][0]["median_traced_call_us"] = 2
    result = comparison(cases)
    assert result["rows"][5]["qualified_speedup_vs_native"] is None
    assert result["rows"][9]["qualified_speedup_vs_native"] is None
    assert result["fastest_qualified"]["traced_call_us"] == 1000
    assert not result["promoted_to_model"]


def test_baseline_drift_disqualifies_all_speed_claims():
    cases = fixture()
    cases[-1]["candidates"][0]["median_traced_call_us"] = 1100
    result = comparison(cases)
    assert not result["timing_comparison_qualified"] and result["fastest_qualified"] is None
    assert all(r["qualified_speedup_vs_native"] is None for r in result["rows"])
    assert all(p["qualified_speedup"] is None for p in result["matched_placement_pairs"])


def test_mismatched_operands_or_work_partition_cannot_be_compared(expect_error):
    cases = fixture()
    for key in ("positions", "query_bf16_sha256", "reference_fp32_sha256", "precision_mode"):
        bad = copy.deepcopy(cases)
        bad[4][key] = "different"
        with expect_error(ValueError, "Unmatched"):
            comparison(bad)
    cases[5]["placement"]["cores_per_user"] -= 1
    with expect_error(ValueError, "partitioning"):
        comparison(cases)


def test_location_only_claim_requires_identical_outputs():
    cases = fixture()
    cases[5]["candidates"][0]["output_fp32_sha256_per_rank"] = ["different"] * 4
    cases[5]["baseline_repeat_output_fp32_sha256_per_rank"] = ["different"] * 4
    pair = comparison(cases)["matched_placement_pairs"][0]
    assert pair["both_numerically_passed"]
    assert not pair["output_bitwise_identical"] and pair["qualified_speedup"] is None
