# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frozen compact GDN coverage, independent of device imports."""

MODES = ((False, False, 0), (False, True, 0), (True, False, 0), (True, True, 0), (True, True, 2560))
CASES = [(b, m, mode) for b in (16, 32) for m in ("l1", "dram") for mode in MODES]
CASES += [(b, m, MODES[-1]) for b in (1, 17, 31) for m in ("l1", "dram")]
BASELINE = "single_step_flat_prepare_epilogue"
CANDIDATE = "single_step_compact_gdn"


def changing_input_checkpoints(updates):
    """Retain the quick integration gate and a separate 4K-token stress gate."""
    if type(updates) is not int or updates not in (64, 4096):
        raise ValueError("Compact GDN comparison requires 64 or 4096 updates")
    return tuple(1 << bit for bit in range(updates.bit_length()))


def validate_long_horizon(report):
    if report.get("state") != "completed" or report.get("cleanup_completed") is not True:
        raise ValueError("Long-horizon comparison did not complete cleanly")
    if (report.get("baseline"), report.get("candidate")) != (BASELINE, CANDIDATE):
        raise ValueError("Long-horizon policies differ from the qualified control and compact candidate")
    if len(report.get("device_ids", [])) != 4 or len(set(report["device_ids"])) != 4:
        raise ValueError("Long-horizon comparison requires four distinct ranks")
    cases = report.get("cases", [])
    if len(cases) != 2 or {c.get("batch") for c in cases} != {16, 32}:
        raise ValueError("Long-horizon comparison lacks B16/B32 coverage")
    for case in cases:
        if (
            case.get("updates") != 4096
            or case.get("all_ranks_bit_identical") is not True
            or case.get("persistent_sessions_precede_trace_capture") is not True
        ):
            raise ValueError("Long-horizon comparison lacks exact independent-session coverage")
        checks = case.get("checkpoints", [])
        if tuple(c.get("step") for c in checks) != changing_input_checkpoints(4096):
            raise ValueError("Long-horizon checkpoints are incomplete")
        for check in checks:
            hashes = check.get("recurrent_conv_projected_sha256", [])
            if (
                check.get("all_values_finite") is not True
                or len(hashes) != 3
                or any(
                    len(ranks) != 4
                    or any(len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in ranks)
                    for ranks in hashes
                )
            ):
                raise ValueError("Long-horizon checkpoint lacks finite four-rank state/history/output evidence")
    return dict(
        updates_per_batch=4096,
        batches=[16, 32],
        exact_boundary_comparison_passed=True,
        independent_dense_reference=False,
        full_model_qualified=False,
        promoted_to_serving=False,
    )
