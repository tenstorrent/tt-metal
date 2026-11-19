# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Delegate the unchanged decoder oracle; --cpu-plan never imports TTNN."""

import argparse
import json
import sys

from models.autoports.qwen_qwen3_6_27b.tt.tiled_batch_prefill_candidate import (
    make_decoder,
    matmul_buffer_bytes,
    matmul_plan,
    row_accounting,
)


def check_buffer_regression():
    policy = {"prefill_grid": (11, 10), "prefill_k": 8, "prefill_subblock_h": True}
    old = matmul_plan(2048, 5120, 8704, policy, {"out_block_h": 7})
    new = matmul_plan(2048, 5120, 8704, policy)
    old_bytes, new_bytes = matmul_buffer_bytes(old), matmul_buffer_bytes(new)
    assert old["per_core_M"] == new["per_core_M"] == 7
    assert old["per_core_N"] == new["per_core_N"] == 25
    assert old_bytes["total"] == 1739776 and new_bytes["total"] == 621568
    assert old_bytes["total"] > 1572864
    # Observed static end includes112576 bytes beyond these four buffers.
    # Preserve this measured offset as a regression check, not a universal ABI.
    assert old_bytes["total"] + 112576 == 1852352
    assert new_bytes["total"] + 112576 < 1572864
    count = 0
    for rows in (64, 1024, 2048, 4096):
        for width, n in ((5120, 8704), (4352, 5120), (1536, 5120), (5120, 3584)):
            plan = matmul_plan(rows, width, n, policy)
            assert plan["out_block_h"] == 1
            assert matmul_buffer_bytes(plan)["total"] <= new_bytes["total"]
            assert plan["per_core_M"] * 10 * 32 >= rows
            count += 1
    joint = matmul_plan(2048, 5120, 7168, policy, {"grid": (8, 8), "out_block_h": 4})
    assert joint["out_block_h"] == 4  # Explicit role-level geometry is retained.
    assert matmul_buffer_bytes(joint, output_tile_bytes=4096)["total"] == 1077248
    adapted = matmul_plan(2048, 5120, 8704, policy, {"grid": (8, 8), "out_block_h": 2})
    assert adapted["per_core_M"] == 8 and adapted["per_core_N"] == 34
    assert matmul_buffer_bytes(adapted)["total"] == 1075200
    return {
        "old_gateup_plan": old,
        "new_gateup_plan": new,
        "old_major_cb_bytes": old_bytes,
        "new_major_cb_bytes": new_bytes,
        "adapted_gateup_plan": adapted,
        "adapted_major_cb_bytes": matmul_buffer_bytes(adapted),
        "extra_role_shape_checks": count,
        "joint_explicit_height_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-plan", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.cpu_plan:
        if remaining:
            raise SystemExit("--cpu-plan accepts no runner flags")
        cases = [row_accounting(b, s) for b in (1, 32) for s in (33, 512, 513)]
        for case in cases:
            case["joint_8x8_plan"] = matmul_plan(
                case["tiled_batch_rows"],
                5120,
                7168,
                {"prefill_grid": (8, 8), "prefill_k": 8, "prefill_subblock_h": True, "prefill_height_block": True},
            )
        print(
            json.dumps(
                {
                    "cases": cases,
                    "hardware_validated": False,
                    "buffer_regression": check_buffer_regression(),
                    "note": "513 rows actually schedule as512+1; table also exposes unsplit padding cost",
                },
                indent=2,
            )
        )
        return
    if "--baseline" in remaining:
        raise SystemExit("Use the original runner for baseline measurements")
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    runner.MultichipDecoder = make_decoder(runner.MultichipDecoder)
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
