# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic A/B control around the unchanged adapter async transition probe."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import SlotStateMover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-state-copies", action="store_true")
    parser.add_argument("--native-rank-sum", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-order", default="immediate-first")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; use a fresh output path")
    if args.eager_state_copies:
        # Reproduce the prior transfer operation sequence, without changing
        # model arithmetic, buffers, decode, sampling or async boundaries.
        SlotStateMover.prepare_copy_traces = lambda self, capture: None
    if args.native_rank_sum:
        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder

        multichip_decoder.exact_rank_sum = lambda value: ttnn.sum(
            ttnn.typecast(value, ttnn.float32), dim=1, keepdim=True
        )
    sys.argv = [sys.argv[0], "--output", str(args.output), "--read-order", args.read_order]
    try:
        runpy.run_module(
            "models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async_transitions", run_name="__main__"
        )
    finally:
        if args.output.exists():
            result = json.loads(args.output.read_text())
            result["state_copy_control"] = "prior_eager_transfers" if args.eager_state_copies else "traced_transfers"
            result["rank_sum_control"] = "native_fp32" if args.native_rank_sum else "custom_exact"
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
