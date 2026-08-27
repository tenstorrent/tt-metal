# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runs unified_kernels/example_matmul.cpp: relu(A @ B + bias) on a 2x2 core grid.

Core (r, c) owns one 2x2-tile block of the output and accumulates over 2 k-blocks.
A is multicast along each grid row, B down each grid column, so every operand tile is
read from DRAM once and shared by the cores that need it.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python example_matmul.py
"""

import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/example_matmul.cpp"
TILE = 32
GRID_H, GRID_W, K_BLOCKS = 2, 2, 2
RT, CT, KT = 2, 2, 2


def to_device(device, t):
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    m, k, n = GRID_H * RT, K_BLOCKS * KT, GRID_W * CT

    torch.manual_seed(0)
    a = (torch.rand([m * TILE, k * TILE]) - 0.5).to(torch.bfloat16)
    b = ((torch.rand([k * TILE, n * TILE]) - 0.5) / (k * TILE) ** 0.5).to(torch.bfloat16)
    bias_row = (torch.rand([n * TILE]) - 0.5).to(torch.bfloat16)

    a_blocks = torch.cat(
        [
            a[r * RT * TILE : (r + 1) * RT * TILE, j * KT * TILE : (j + 1) * KT * TILE]
            for r in range(GRID_H)
            for j in range(K_BLOCKS)
        ]
    )
    b_blocks = torch.cat(
        [
            b[j * KT * TILE : (j + 1) * KT * TILE, c * CT * TILE : (c + 1) * CT * TILE]
            for c in range(GRID_W)
            for j in range(K_BLOCKS)
        ]
    )
    bias = bias_row.repeat(TILE, 1)

    device = ttnn.open_device(device_id=0)
    try:
        ta, tb = to_device(device, a_blocks), to_device(device, b_blocks)
        tbias = to_device(device, bias)
        tout = to_device(device, torch.full([GRID_H * GRID_W * RT * TILE, CT * TILE], float("nan")))

        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_W - 1, GRID_H - 1))])
        cores = [ttnn.CoreCoord(x, y) for y in range(GRID_H) for x in range(GRID_W)]

        tensors = {"a": ta, "b": tb, "bias": tbias, "out": tout}
        spec = unified_program_spec(
            kernel_source=KERNEL,
            nodes=core_ranges,
            dfbs=[
                dfb("a", 2 * RT * KT),
                dfb("b", 2 * KT * CT),
                dfb("bias", CT),
                dfb("out", RT * CT),
                dfb("partials", RT * CT),
            ],
            tensors=tensors,
            name="example_matmul",
        )
        run_unified_spec(device, spec, tensors)
        out = tout
        got = ttnn.to_torch(out).to(torch.float32)[0, 0]
    finally:
        ttnn.close_device(device)

    want = torch.relu(a.to(torch.float32) @ b.to(torch.float32) + bias_row.to(torch.float32))
    want_blocks = torch.cat(
        [
            want[r * RT * TILE : (r + 1) * RT * TILE, c * CT * TILE : (c + 1) * CT * TILE]
            for r in range(GRID_H)
            for c in range(GRID_W)
        ]
    )

    error = (got - want_blocks).abs().max().item()
    logger.info(f"relu(A @ B + bias) on a {GRID_H}x{GRID_W} grid: max|error| = {error:.5f}")
    if error > 0.01:
        logger.error("FAIL")
        return 1
    logger.info("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
