# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched-location and chunk-size comparisons, without device imports."""

import statistics

from models.demos.qwen38_27b_qb2.tests.attention_placement import useful_kv_bytes

# Prioritize the highest useful operating points. Keep 32K tradeoffs visible.
CASES = ((262016, 8), (131072, 16), (262016, 4), (131072, 8), (32768, 16), (32768, 32))
VARIANTS = (
    ("native", 256),
    ("full_grid_sharded", 256),
    ("native", 512),
    ("full_grid_sharded", 512),
    ("row_major_80", 256),
    ("outer_columns_80", 256),
    ("row_major_80", 512),
    ("outer_columns_80", 512),
    ("row_major_96", 256),
    ("outer_columns_96", 256),
    ("row_major_96", 512),
    ("outer_columns_96", 512),
    ("native", 256),
)


def comparison(cases):
    """Never count a failed numerical/timing candidate as a speedup."""
    if len(cases) != len(VARIANTS):
        raise ValueError("Expected the complete bracketed placement/chunk experiment")
    first = cases[0]
    matched_keys = (
        "input_tokens",
        "batch",
        "positions",
        "seed",
        "page_table_sha256",
        "query_bf16_sha256",
        "reference_fp32_sha256",
        "precision_mode",
        "worker_grid",
    )
    rows = []
    for case, (name, chunk) in zip(cases, VARIANTS):
        if any(case[key] != first[key] for key in matched_keys):
            raise ValueError("Unmatched operands, geometry or precision in placement comparison")
        if case["placement"]["name"] != name or len(case["candidates"]) != 1:
            raise ValueError("Unexpected placement variant/order")
        candidate = case["candidates"][0]
        if candidate["chunk"] != chunk:
            raise ValueError("Unexpected chunk variant/order")
        repeated_identically = (
            candidate["output_fp32_sha256_per_rank"] == case["baseline_repeat_output_fp32_sha256_per_rank"]
        )
        passed = case["passed"] and candidate["accuracy_passed"] and repeated_identically
        qualified = passed and case["selection"]["timing_comparison_qualified"]
        us = candidate["median_traced_call_us"]
        rows.append(
            dict(
                name=name,
                chunk=chunk,
                traced_call_us=us,
                numerically_passed=passed,
                deterministic_replay=repeated_identically,
                timing_qualified=qualified,
                useful_kv_gb_s=useful_kv_bytes(case["positions"]) / (us * 1000),
                cores_per_user=case["placement"]["cores_per_user"],
                active_cores=case["placement"]["active_cores"],
                output_hashes=candidate["output_fp32_sha256_per_rank"],
            )
        )
    if not rows[0]["numerically_passed"] or not rows[-1]["numerically_passed"]:
        raise ValueError("Native accurate-attention control failed")
    if rows[0]["output_hashes"] != rows[-1]["output_hashes"]:
        raise ValueError("Native control changed output across the experiment")
    bracket_drift = abs(rows[-1]["traced_call_us"] / rows[0]["traced_call_us"] - 1)
    native_us = statistics.mean((rows[0]["traced_call_us"], rows[-1]["traced_call_us"]))
    bracket_qualified = bracket_drift <= 0.03 and rows[0]["timing_qualified"] and rows[-1]["timing_qualified"]
    for row in rows:
        row["timing_qualified"] &= bracket_qualified
        row["qualified_speedup_vs_native"] = native_us / row["traced_call_us"] if row["timing_qualified"] else None
    pairs = []
    for chunk in (256, 512):
        for count in (80, 96):
            a, b = (
                next(r for r in rows if r["name"] == name and r["chunk"] == chunk)
                for name in (f"row_major_{count}", f"outer_columns_{count}")
            )
            if (a["active_cores"], a["cores_per_user"]) != (b["active_cores"], b["cores_per_user"]):
                raise ValueError("Matched-placement pair changes work partitioning")
            pairs.append(
                dict(
                    cores=count,
                    chunk=chunk,
                    control=a["name"],
                    candidate=b["name"],
                    output_bitwise_identical=a["output_hashes"] == b["output_hashes"],
                    both_numerically_passed=a["numerically_passed"] and b["numerically_passed"],
                    qualified_speedup=a["traced_call_us"] / b["traced_call_us"]
                    if a["timing_qualified"] and b["timing_qualified"] and a["output_hashes"] == b["output_hashes"]
                    else None,
                )
            )
    valid = [r for r in rows[:-1] if r["timing_qualified"]]
    return dict(
        input_tokens=first["input_tokens"],
        batch=first["batch"],
        baseline_repeat_drift_fraction=bracket_drift,
        timing_comparison_qualified=bracket_qualified,
        fastest_qualified=min(valid, key=lambda r: r["traced_call_us"]) if valid else None,
        rows=rows,
        matched_placement_pairs=pairs,
        promoted_to_model=False,
        scope="Synthetic attention only; common DRAM output boundary includes required conversion",
    )
