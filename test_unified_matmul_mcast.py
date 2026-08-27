# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-core matmul with BOTH operands multicast, on an R x C grid of cores.

Core (r, c) computes A_block[r] @ B_block[c]. The LHS is broadcast along each
row, the RHS down each column, using the two harness-reserved handshake pairs.

Operands are block-major so each core's block is contiguous pages: A is R blocks
of rt x kt tiles, B is C blocks of kt x ct tiles, output is R*C blocks of rt x ct
in row-major core order.

    python test_unified_matmul_mcast.py --grid-h 2 --grid-w 2 --rt 2 --ct 2 --kt 2
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, unified_program

KERNEL = "unified_kernels/matmul_mcast.cpp"
TILE = 32


def run(device, grid_h=2, grid_w=2, rt=2, ct=2, kt=2, k_blocks=1, mode="dst", in1_thread=0, seed=0):
    torch.manual_seed(seed)
    # K blocks per core row / column, laid out block-major as r*K + k and c*K + k.
    a_blocks = [(torch.rand([1, 1, rt * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(grid_h * k_blocks)]
    b_blocks = [(torch.rand([1, 1, kt * TILE, ct * TILE]) - 0.5).to(torch.bfloat16) for _ in range(grid_w * k_blocks)]
    a = torch.cat(a_blocks, dim=2)
    b = torch.cat(b_blocks, dim=2)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, grid_h * grid_w * rt * TILE, ct * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_h) for x in range(grid_w)]

    # Same args on every core: each one works out its own row/column on device.

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        cbs=[
            make_cb(CB_IN0, core_ranges, num_pages=rt * kt),
            make_cb(CB_IN1, core_ranges, num_pages=kt * ct),
            # Partials, exactly one block: L1 mode relies on push/pop wrapping it.
            make_cb(CB_ACC, core_ranges, num_pages=rt * ct),
            make_cb(CB_OUT, core_ranges, num_pages=rt * ct),
        ],
        tensors={"in0": ta, "in1": tb, "out": tout},
        defines=[
            ("MM_RT", str(rt)),
            ("MM_CT", str(ct)),
            ("MM_KT", str(kt)),
            ("MM_GRID_H", str(grid_h)),
            ("MM_GRID_W", str(grid_w)),
            ("MM_K_BLOCKS", str(k_blocks)),
            # 1 on hardware for a second NOC; ttsim cannot multicast on NOC 1.
            ("MM_IN1_THREAD", str(in1_thread)),
        ]
        + ([("MM_ACC_L1", "1")] if mode == "l1" else []),
    )

    logger.info(
        f"running unified matmul mcast: grid={grid_h}x{grid_w} rt={rt} ct={ct} kt={kt} "
        f"k_blocks={k_blocks} mode={mode}"
    )
    run_unified_spec(device, spec, {"in0": ta, "in1": tb, "out": tout})
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)
    blocks = []
    for r in range(grid_h):
        for c in range(grid_w):
            acc = torch.zeros([1, 1, rt * TILE, ct * TILE], dtype=torch.float32)
            for k in range(k_blocks):
                acc = acc + a_blocks[r * k_blocks + k].to(torch.float32) @ b_blocks[c * k_blocks + k].to(torch.float32)
            blocks.append(acc)
    want = torch.cat(blocks, dim=2)
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--grid-h", type=int, default=2, help="core rows (LHS broadcast spans a row)")
    p.add_argument("--grid-w", type=int, default=2, help="core columns (RHS broadcast spans a column)")
    p.add_argument("--rt", type=int, default=2)
    p.add_argument("--ct", type=int, default=2)
    p.add_argument("--kt", type=int, default=2)
    p.add_argument("--k-blocks", type=int, default=1, help="k-blocks each core accumulates over")
    p.add_argument("--mode", choices=["dst", "l1"], default="dst")
    p.add_argument("--in1-thread", type=int, default=0, choices=[0, 1], help="DM thread for the RHS broadcast")
    p.add_argument("--pcc", type=float, default=0.99)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(
            device, args.grid_h, args.grid_w, args.rt, args.ct, args.kt, args.k_blocks, args.mode, args.in1_thread
        )
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    # Per-core-block error, so a single mis-fed core is visible rather than averaged away.
    rows = args.rt * TILE
    worst = 0.0
    for i in range(args.grid_h * args.grid_w):
        g = got[0, 0, i * rows : (i + 1) * rows, :]
        w = want[0, 0, i * rows : (i + 1) * rows, :]
        d = (g - w).abs().max().item()
        worst = max(worst, d)
        logger.info(f"  core block {i} (row {i // args.grid_w}, col {i % args.grid_w}): max|err| = {d:.4f}")
    logger.info(f"worst per-core max|err| = {worst:.4f}")
    if measured < args.pcc:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
