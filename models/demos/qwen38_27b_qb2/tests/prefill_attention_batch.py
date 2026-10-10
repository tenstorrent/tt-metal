# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frozen physical coverage and acceptance for batched paged-prefill attention."""

import math
import statistics

CASES = ((2, 0, 128), (3, 32, 65), (16, 0, 2048), (16, 14336, 2048), (16, 30720, 2048), (32, 31744, 1024))


def validate_report(report):
    if (
        report.get("state") != "completed"
        or report.get("passed") is not True
        or report.get("cleanup_completed") is not True
    ):
        raise ValueError("Prefill boundary lacks complete clean hardware evidence")
    cases = report["cases"]
    if [(r["batch"], r["start_pos"], r["chunk_tokens"]) for r in cases] != list(CASES):
        raise ValueError("Prefill coverage differs from frozen plan")
    output = []
    for row in cases:
        arms = row["arms"]
        if [arm["name"] for arm in arms] != ["before", "batched", "after"]:
            raise ValueError("Require before/batched/after prefill controls")
        for arm in arms:
            times = arm["wall_ms"]
            if (
                len(times) != 5
                or any(not math.isfinite(t) or t <= 0 for t in times)
                or statistics.median(times) != arm["median_ms"]
            ):
                raise ValueError("Invalid prefill measurement accounting")
            if len(arm["output_sha256_per_rank"]) != 4 or len(set(arm["output_sha256_per_rank"])) != 1:
                raise ValueError("All four replicated ranks must agree")
            if len(row["expected_cache_sha256"]) != 2 or arm["cache_sha256_per_rank"] != [
                [digest] * 4 for digest in row["expected_cache_sha256"]
            ]:
                raise ValueError("Cache contents differ from expected ownership and writes")
            checks = arm["accuracy_per_rank"]
            if len(checks) != 4 or any(
                check.get("passed") is not True
                or len(check.get("pcc_per_user", [])) != row["batch"]
                or len(check.get("relative_rms_per_user", [])) != row["batch"]
                or any(not math.isfinite(v) or v < 0.999 for v in check.get("pcc_per_user", []))
                or any(not math.isfinite(v) or not 0 <= v <= 0.02 for v in check.get("relative_rms_per_user", []))
                for check in checks
            ):
                raise ValueError("Every rank and user needs dense reference checks")
        before, candidate, after = arms
        if (
            before["output_sha256_per_rank"] != candidate["output_sha256_per_rank"]
            or before["output_sha256_per_rank"] != after["output_sha256_per_rank"]
        ):
            raise ValueError("Batched prefill output differs from controls")
        drift = abs(after["median_ms"] / before["median_ms"] - 1)
        output.append(
            dict(
                batch=row["batch"],
                context=row["total_context"],
                chunk_tokens=row["chunk_tokens"],
                control_ms=statistics.median([before["median_ms"], after["median_ms"]]),
                candidate_ms=candidate["median_ms"],
                control_drift_fraction=drift,
                timing_qualified=drift <= 0.03,
                speedup=statistics.median([before["median_ms"], after["median_ms"]]) / candidate["median_ms"]
                if drift <= 0.03
                else None,
                full_model_measured=False,
                gpqa_qualified=False,
            )
        )
    return output
