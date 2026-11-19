# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reuse the strict production-layout scan oracle for isolated coefficient caching.

No hardware unless --run-device is explicit. All mode, trace, real-position227,
coefficient/product/partial/final equality gates are inherited without weakening.
"""

import ast
import json
from pathlib import Path

from . import run_stage7_scan_probe as base

CACHE_VARIANTS = {
    "cache": (2048, "baseline"),
    "cache32": (32, "baseline"),
    "cache64": (64, "baseline"),
    "cache32_nop": (32, "nop"),
    "cache32_elide": (32, "elide"),
    "cache32_noclear": (32, "baseline"),
    "cache32_poison": (32, "baseline"),
    "cache_noclear": (2048, "baseline"),
    "cache64_noclear": (64, "baseline"),
}


def schedule_proof():
    # Execute only the current pure partition function, avoiding TTNN imports in CPU mode.
    tree = ast.parse((base.ROOT / "tt/fused_decoder.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "terminal_projection_schedule")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "current_terminal_projection_schedule", "exec"), scope)
    schedule = scope["terminal_projection_schedule"]
    records = []
    for role, blocks, ht, wt in (
        ("input", 4, 16, 98),
        ("output", 2, 16, 64),
        ("router", 4, 16, 8),
        ("input1024", 2, 32, 98),
    ):
        groups = 0
        for x, y, start, count in schedule(blocks * wt, 11, 10):
            produced = [
                run // wt for run in range(start, start + count) if run == start or run // wt != (run - 1) // wt
            ]
            consumed = [
                run // wt
                for run in range(start, start + count)
                if run + 1 == start + count or (run + 1) // wt != run // wt
            ]
            assert produced == consumed
            for block in produced:
                indices = [h for run in range(start, start + count) if run // wt == block for h in range(ht)]
                assert len(indices) % ht == 0 and set(indices) == set(range(ht))
            groups += len(produced)
        records.append(
            dict(
                role=role,
                grid=[11, 10],
                old_coefficient_fills=blocks * wt * ht,
                cached_coefficient_fills=groups * ht,
                cache_bytes_per_worker=ht * 2048,
                extra_cb_bytes_per_worker=(ht - 2) * 2048,
            )
        )
    return records


def main():
    original_run = base.run
    base.VARIANTS.clear()
    base.VARIANTS.update(CACHE_VARIANTS)

    def run(args):
        assert not args.output.exists(), "Preserve prior evidence"
        proof = schedule_proof()
        padding = (
            "none" if args.variant.endswith("_noclear") else "poison" if args.variant.endswith("_poison") else "zero"
        )
        valid_lanes = [(row // 16) * 512 + (row % 16) * 16 for row in range(32)]
        assert len(set(valid_lanes)) == 32 and all(0 <= lane < 1024 for lane in valid_lanes)
        helper_module = original_class = None
        try:
            if args.run_device:
                # Import only after the same environment preconditions as the base
                # harness, since TTNN snapshots allocation tracking at import.
                import os

                assert args.timing and 227 in args.positions
                assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
                assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
                from models.autoports.qwen_qwen3_6_35b_a3b.tt import scan_terminal_probe as helper_module
                from models.autoports.qwen_qwen3_6_35b_a3b.tt.scan_coefficient_cache_probe import CachedCoefficientScan

                original_class = helper_module.ScanTerminalProbe
                from functools import partial

                helper_module.ScanTerminalProbe = partial(CachedCoefficientScan, padding=padding)
            original_run(args)
        finally:
            if helper_module is not None and original_class is not None:
                helper_module.ScanTerminalProbe = original_class
            if args.output.exists():
                record = json.loads(args.output.read_text())
                record["coefficient_cache"] = {
                    "scope": "Run-major per-worker K-block cache; arithmetic and run schedule unchanged; CB1 retained through final local run then popped in full",
                    "schedule_proof": proof,
                    "padding_mode": padding,
                    "valid_coefficient_bf16_offsets": valid_lanes,
                    "physical_padding_contract": "Only column0 (32 lanes/tile) is logical and broadcast-consumed; other992 BF16 lanes are unspecified in no-clear mode or qNaN0x7fc1 in poison mode. Full product/partial/final gates unchanged.",
                    "l1_headroom_full_model_verified": False,
                    "base_harness": str(Path(base.__file__).resolve()),
                    "source_sha256": {
                        str(p.relative_to(base.REPO)): base.sha(p)
                        for p in (Path(__file__), base.ROOT / "tt/scan_coefficient_cache_probe.py")
                    },
                }
                args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    base.run = run
    base.main()


if __name__ == "__main__":
    main()
