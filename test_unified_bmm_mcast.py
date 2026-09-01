# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched multi-core matmul with BOTH operands multicast, on an R x C grid of cores.

matmul_mcast plus a batch dimension. Core (r, c) computes A[n][r] @ B[n][c] for every
batch n, with the LHS broadcast along each row and the RHS down each column using the two
harness-reserved handshake pairs.

Operands are block-major so each core's block is contiguous pages:

    A   batch*R*K blocks of rt x kt   indexed (n*R + r)*K + k
    B   batch*C*K blocks of kt x ct   indexed (n*C + c)*K + k
    out batch*R*C blocks of rt x ct   indexed  n*R*C + r*C + c

Reported PER (batch, core) BLOCK as well as by PCC, because the PATTERN of the error is
what names the bug. Under multicast an indexing mistake is never confined to one block: a
receiver core never reads the tensor at all, so only SENDER indices are load-bearing, and
corrupting one sender takes out its whole row or column. Measured -- mis-indexing column
1's sender for batch 1 gives max|err| 5.1 and 5.2 on exactly the two blocks of that column,
PCC 0.76, everything else clean.

So --block-tol is a tripwire and a diagnostic, not a check that catches something PCC
misses; at these sizes PCC caught every deliberate bug tried. The per-block table is what
tells you WHICH sender, from the shape of the failures.

    python test_unified_bmm_mcast.py --batch 2 --grid-h 2 --grid-w 2 --k-blocks 2
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/bmm_mcast.cpp"
TILE = 32


def run(device, batch=2, grid_h=2, grid_w=2, rt=2, ct=2, kt=2, k_blocks=2, mode="dst", in1_thread=0, seed=0):
    torch.manual_seed(seed)
    # Block-major, batch outermost: index (n*R + r)*K + k and (n*C + c)*K + k.
    a_blocks = [
        (torch.rand([1, 1, rt * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(batch * grid_h * k_blocks)
    ]
    b_blocks = [
        (torch.rand([1, 1, kt * TILE, ct * TILE]) - 0.5).to(torch.bfloat16) for _ in range(batch * grid_w * k_blocks)
    ]
    a = torch.cat(a_blocks, dim=2)
    b = torch.cat(b_blocks, dim=2)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, batch * grid_h * grid_w * rt * TILE, ct * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram,
    )

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))])

    # Same args on every core: each one works out its own row/column on device.
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb("in0", rt * kt),
            dfb("in1", kt * ct),
            dfb("acc", rt * ct),
            dfb("out", rt * ct),
        ],
        tensors={"in0": ta, "in1": tb, "out": tout},
        named_compile_time_args=[
            ("batch", batch),
            ("rt", rt),
            ("ct", ct),
            ("kt", kt),
            ("k_blocks", k_blocks),
            ("grid_h", grid_h),
            ("grid_w", grid_w),
        ],
        # Only the two knobs that cannot be named args: ACC_L1 picks an Accumulator type,
        # and IN1_THREAD is a noc_load thread the harness resolves from defines.
        defines=[
            # 1 on hardware for a second NOC; ttsim cannot multicast on NOC 1.
            ("BMM_IN1_THREAD", str(in1_thread)),
        ]
        + ([("BMM_ACC_L1", "1")] if mode == "l1" else []),
    )

    logger.info(
        f"running unified bmm mcast: batch={batch} grid={grid_h}x{grid_w} rt={rt} ct={ct} kt={kt} "
        f"k_blocks={k_blocks} mode={mode}"
    )
    run_unified_spec(device, spec, {"in0": ta, "in1": tb, "out": tout})

    got = ttnn.to_torch(tout).to(torch.float32)
    blocks = []
    for n in range(batch):
        for r in range(grid_h):
            for c in range(grid_w):
                acc = torch.zeros([1, 1, rt * TILE, ct * TILE], dtype=torch.float32)
                for k in range(k_blocks):
                    ai = (n * grid_h + r) * k_blocks + k
                    bi = (n * grid_w + c) * k_blocks + k
                    acc = acc + a_blocks[ai].to(torch.float32) @ b_blocks[bi].to(torch.float32)
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
    p.add_argument("--batch", type=int, default=2, help="batches each core walks")
    p.add_argument("--grid-h", type=int, default=2, help="core rows (LHS broadcast spans a row)")
    p.add_argument("--grid-w", type=int, default=2, help="core columns (RHS broadcast spans a column)")
    p.add_argument("--rt", type=int, default=2)
    p.add_argument("--ct", type=int, default=2)
    p.add_argument("--kt", type=int, default=2)
    p.add_argument("--k-blocks", type=int, default=2, help="k-blocks each core accumulates over, per batch")
    p.add_argument("--mode", choices=["dst", "l1"], default="dst")
    p.add_argument("--in1-thread", type=int, default=0, choices=[0, 1], help="DM thread for the RHS broadcast")
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument(
        "--block-tol",
        type=float,
        default=0.5,
        help="max|err| allowed in ANY single (batch, core) block; catches index arithmetic PCC hides",
    )
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(
            device,
            args.batch,
            args.grid_h,
            args.grid_w,
            args.rt,
            args.ct,
            args.kt,
            args.k_blocks,
            args.mode,
            args.in1_thread,
        )
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")

    # Per (batch, core) block, so a single mis-fed core is visible rather than averaged away.
    rows = args.rt * TILE
    worst = 0.0
    per_batch_worst = []
    for n in range(args.batch):
        bw = 0.0
        for i in range(args.grid_h * args.grid_w):
            blk = n * args.grid_h * args.grid_w + i
            g = got[0, 0, blk * rows : (blk + 1) * rows, :]
            w = want[0, 0, blk * rows : (blk + 1) * rows, :]
            d = (g - w).abs().max().item()
            bw = max(bw, d)
            logger.info(
                f"  batch {n} core block {i} (row {i // args.grid_w}, col {i % args.grid_w}): max|err| = {d:.4f}"
            )
        per_batch_worst.append(bw)
        worst = max(worst, bw)
    logger.info(f"worst per-block max|err| = {worst:.4f}   per batch: {[f'{v:.4f}' for v in per_batch_worst]}")

    # An independent tripwire on the worst single block. PCC caught every deliberate bug
    # tried here, so this is belt-and-braces -- but it fails loudly on a bad block rather
    # than leaving a correlation to be interpreted.
    if worst > args.block_tol:
        logger.error(f"FAIL: a block exceeded --block-tol {args.block_tol} (worst {worst:.4f})")
        return 1
    if measured < args.pcc:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
