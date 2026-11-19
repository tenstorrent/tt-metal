# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the original exact-output/changed-input trace gate on parallel K blocks.

Arguments are those of probe_ordered_projection, plus --compare-original.
That option first runs the serial implementation in its own device lifetime,
then the candidate on the identical inputs. Both must equal the same CPU
ordered BF16 tensors exactly; the candidate also supports --diagnostic-fp32
to check every pre-cast FP32 bit. Use --counts all for final component coverage.
No device is opened merely by importing this module.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_ordered_projection
from models.autoports.qwen_qwen3_8_flash_next.tt import ordered_projection
from models.autoports.qwen_qwen3_8_flash_next.tt.parallel_ordered_projection import parallel_ordered_indexer_projection


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-original", action="store_true")
    args, _ = parser.parse_known_args()
    if args.compare_original:
        sys.argv.remove("--compare-original")
        baseline_path = args.output.with_name(args.output.stem + "_original.json")
        baseline_args = list(sys.argv)
        if "--output" in baseline_args:
            baseline_args[baseline_args.index("--output") + 1] = str(baseline_path)
        else:
            baseline_args = [
                "--output=" + str(baseline_path) if item.startswith("--output=") else item for item in baseline_args
            ]
        with patch.object(sys, "argv", baseline_args):
            probe_ordered_projection.main()
    calls = []

    def selected(value, weight, **kwargs):
        calls.append(int(value.shape[-2]))
        # Match the explicit parallel-indexer-rows integration policy, rather
        # than relying on defaults that a wrapper may override with False.
        kwargs["valid_row_compute"] = True
        kwargs["copy_once"] = True
        return parallel_ordered_indexer_projection(value, weight, **kwargs)

    with patch.object(ordered_projection, "ordered_indexer_projection", selected):
        probe_ordered_projection.main()
    report = json.loads(args.output.read_text())
    if "--cpu-only" not in sys.argv:
        assert calls, "Parallel candidate was never executed"
    report["parallel_projection"] = {
        "logical_call_counts": calls,
        "partial_policy": "Q1:5x512, 100 cores; Q2..32:1024+1024+512, 60 cores",
        "combine": "20 cores; UnpackToDestFp32; ordered SFPU addition; explicit BF16 typecast",
        "valid_row_compute": True,
        "copy_once": True,
        "original_comparison": str(baseline_path) if args.compare_original else None,
    }
    # The inherited probe records FP32 diagnostics but gates only BF16 output.
    # Here the lossless partial boundary is part of the candidate contract:
    # make --diagnostic-fp32 fail on even one changed pre-cast FP32 bit.
    fp32_checks = [row["fp32_diagnostic"] for row in report["device"] if "fp32_diagnostic" in row]
    if "--diagnostic-fp32" in sys.argv and "--cpu-only" not in sys.argv:
        exact_fp32 = bool(fp32_checks) and all(row["different_fp32_bits"] == 0 for row in fp32_checks)
        report["parallel_projection"]["fp32_exact"] = exact_fp32
        report["passed"] = bool(report["passed"] and exact_fp32)
    if args.compare_original:
        baseline = json.loads(baseline_path.read_text())
        assert baseline["cpu"] == report["cpu"], "Original and candidate did not use identical CPU controls"
        report["parallel_projection"]["both_bit_exact_to_identical_control"] = bool(
            baseline["passed"] and report["passed"]
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if "--cpu-only" not in sys.argv:
        assert report["passed"], "Parallel projection failed its exact FP32/BF16 contract; inspect saved diagnostics"


if __name__ == "__main__":
    main()
