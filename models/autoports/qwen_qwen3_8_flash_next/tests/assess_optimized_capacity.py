# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assess the owner's whole-output contract without rewriting legacy probe results.

The existing PLE probe additionally gates every residual-subtracted row at .995.
This stage's explicit contract makes per-row PCC diagnostic. We retain that raw
strict result and require an independent matched HF packed-lookup control when
the extra contribution-row gate misses. Literal whole outputs, all decode rows,
finite outputs, runtime audit and exact reset/replay checks remain hard gates.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--control-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.device_report.read_text())
    phases = [raw["prefill"], *raw["decode"]]
    assert raw["decode_steps"] == 3 and len(raw["decode"]) == 3
    checks = {
        "finite": all(value["finite"] for value in phases),
        "whole_output_pcc_0_995": all(value["whole"]["global_pcc"] >= 0.995 for value in phases),
        "whole_row_pcc_0_995": all(value["whole"]["minimum_row_pcc"] >= 0.995 for value in phases),
        "aggregate_contribution_pcc_0_995": all(value["contribution"]["global_pcc"] >= 0.995 for value in phases),
        "contribution_diagnostic_floor_0_99": all(value["contribution"]["minimum_row_pcc"] >= 0.99 for value in phases),
        "decode_strict": all(value["passed"] for value in raw["decode"]),
        "reset_replay_equal": raw["reset_replay_equal"] is True,
        "reset_state_equal": all(raw["reset_state_equal"].values()),
        "prefill_reset_equal": raw["prefill_reset_equal"] is True,
        "runtime_audit": all(raw["runtime_audit"].values()),
        "trace_exercised": raw["captures"] == 1 and raw["replays"] >= 6,
    }
    control = None
    if not raw["passed"]:
        assert args.control_report, "A failed legacy result requires an independent matched control"
        control = json.loads(args.control_report.read_text())
        assert control["device_cache_identity"] == raw["device_output_cache"]["identity"]
        checks["independent_literal_repeat_exact"] = control["independent_literal_repeat_exact"] is True
        checks["matched_packed_control_all_rows"] = control["matched_device"]["packed"]["passed"] is True
        # No whole-output, decode, runtime, or determinism failure is classifiable.
        assert raw["prefill"]["whole"]["passed"] and not raw["prefill"]["contribution"]["passed"]
    result = dict(
        device_report=str(args.device_report),
        control_report=str(args.control_report) if args.control_report else None,
        acceptance_contract="User: whole-output PCC >= .995; per-row PCC diagnostic. No original report is changed.",
        legacy_strict_passed=raw["passed"],
        checks=checks,
        passed=all(checks.values()),
        prefill=raw["prefill"],
        decode=raw["decode"],
        matched_packed_prefill=control["matched_device"]["packed"] if control else None,
        interpretation="A contribution-row diagnostic is not a failure of unchanged literal HF settings.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], checks
    print(json.dumps(dict(passed=True, legacy_strict_passed=raw["passed"], checks=checks)))


if __name__ == "__main__":
    main()
