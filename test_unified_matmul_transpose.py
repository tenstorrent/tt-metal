# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A REAL A @ B-transpose on device, which takes both halves of the transpose.

Metal's matmul flag transposes each 32x32 tile of B and leaves the tile GRID alone --
"transpose operation on tiles in B" in its own words. So the flag by itself computes
neither A@B nor A@B.T once B is wider than one tile. A true transpose needs:

    per-tile    the hardware flag           (u::TransposeB::Yes)
    tile grid   whoever fills the buffer    (here the host; in a real kernel, the reader)

This test supplies the grid half itself: it builds B as a ct x kt tile grid, moves tile
(c, k) to slot (k, c) without touching any tile's contents, and hands that to the
kernel. The result must equal torch's a @ b.T exactly.

The single-tile case is deliberately included and deliberately NOT the whole test: at
one tile a grid transpose is the identity, so per-tile-only and a true transpose agree,
and a test that stopped there would pass an implementation that never grid-transposes.
Multi-tile shapes are what separate them.

WHAT THIS DOES AND DOES NOT COVER. The flag reaches four places: matmul_block itself,
and three matmul_block_init restores. Verified by forcing each to transpose=0 and
checking which rows below fail:

  matmul_block                      every transposed row fails
  Dst reload restore                ONLY k_blocks>=2 in Dst mode fails -- all five
                                    single-block rows pass, so a single-block suite
                                    would have shipped that bug
  bias_finish restore               NOTHING fails
  L1 biased-finish restore          NOTHING fails

The last two are not gaps in this test; they are unreachable. Both exist to put the FPU
back "for the next output block", and no kernel in this repo emits more than one output
block per launch -- each does a single acc.clear() and one k-loop. Their transpose
argument is therefore correct by construction and unverified by execution. The first
kernel that loops over output blocks (attention, over Q chunks) makes them live, and
they must be re-verified then.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_matmul_transpose.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/matmul.cpp"
TILE = 32


def grid_transpose(b, ct, kt):
    """Move tile (c, k) to slot (k, c). Tile CONTENTS are untouched -- the hardware
    flag does the within-tile half."""
    v = b.reshape(ct, TILE, kt, TILE)  # c, i, k, j
    return v.permute(2, 1, 0, 3).reshape(kt * TILE, ct * TILE)  # k, i, c, j


def run(device, rt, ct, kt, k_blocks=1, mode="dst", bias=False, seed=0):
    torch.manual_seed(seed)
    # A is rt x kt tiles per block; B is ct x kt tiles per block, because it is the
    # operand we transpose.
    a_blocks = [(torch.rand([rt * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]
    b_blocks = [(torch.rand([ct * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]

    a = torch.cat(a_blocks, dim=0)
    # Each block is grid-transposed independently: the kernel walks one block per k step.
    b_dev = torch.cat([grid_transpose(x.to(torch.float32), ct, kt).to(torch.bfloat16) for x in b_blocks], dim=0)

    # A bias reaches a restore site nothing else here does: in L1 mode the biased
    # finish is a separate branch with its own matmul_block_init afterwards.
    # Replicated down all 32 rows of each tile, which is what bias() requires: two of the
    # three paths add it with an elementwise FPU dest-reuse add and read every row. Leaving
    # rows 1..31 zeroed -- as this test did -- applies the bias to one output row in 32.
    bias_row = ((torch.rand([ct * TILE]) - 0.5) * 4.0).to(torch.bfloat16)
    bias_t = torch.zeros([1, 1, TILE, ct * TILE], dtype=torch.bfloat16)
    bias_t[0, 0, :, :] = bias_row

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(
        a.reshape(1, 1, *a.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )
    tb = ttnn.from_torch(
        b_dev.reshape(1, 1, *b_dev.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
    )
    tbias = ttnn.from_torch(bias_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rt * TILE, ct * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges, cores = single_core()
    # bias accessor args go LAST, matching the kernel's arg layout.
    dfbs = [
        dfb("in0", rt * kt),
        dfb("in1", kt * ct),
        dfb("acc", rt * ct),
        dfb("out", rt * ct),
        # Declared even without MM_BIAS: the kernel declares its Storage unconditionally.
        dfb("bias", ct),
    ]
    # The bias tensor is always bound too, for the same reason -- the kernel names
    # tensor::bias on every projection whether or not the fusion reads it.
    tensors = {"in0": ta, "in1": tb, "out": tout, "bias": tbias}
    defines = (
        [
            ("MM_RT_DIM", str(rt)),
            ("MM_CT_DIM", str(ct)),
            ("MM_KT_DIM", str(kt)),
            ("MM_K_BLOCKS", str(k_blocks)),
            ("MM_TRANSPOSE", "1"),
        ]
        + ([("MM_ACC_L1", "1")] if mode == "l1" else [])
        + ([("MM_BIAS", "1")] if bias else [])
    )

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors=tensors,
        defines=defines,
        name="matmul_transpose",
    )
    run_unified_spec(device, spec, tensors)
    out = tout
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]

    # The truth: a real transpose, summed over the k blocks.
    want = torch.zeros([rt * TILE, ct * TILE], dtype=torch.float32)
    for x, y in zip(a_blocks, b_blocks):
        want += x.to(torch.float32) @ y.to(torch.float32).T
    if bias:
        want += bias_row.to(torch.float32).unsqueeze(0)  # broadcast down the rows
    return got, want


def err(got, want):
    """Normalised by the tensor's own scale: these are zero-centred products, so an
    elementwise denominator goes to zero and says nothing."""
    return ((got - want).abs().max() / want.abs().max()).item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=0.02)
    args = p.parse_args(argv)

    # (rt, ct, kt, k_blocks, mode). 1x1x1 is the degenerate case; the rest separate a
    # true transpose from a per-tile-only one, and the multi-block rows exercise the
    # matmul_block_init restores in each accumulator mode.
    # (rt, ct, kt, k_blocks, mode, bias). The bias rows are not decoration: in L1 mode
    # the biased finish is a separate branch with its own matmul_block_init restore, and
    # a transpose missing from THAT one is invisible to every row without a bias.
    cases = [
        (1, 1, 1, 1, "dst", False),
        (1, 2, 2, 1, "dst", False),
        (2, 2, 2, 1, "dst", False),
        (2, 4, 2, 1, "dst", False),
        (1, 4, 2, 1, "dst", False),
        (2, 2, 2, 2, "dst", False),
        (2, 2, 2, 2, "l1", False),
        (2, 4, 2, 3, "l1", False),
        (2, 2, 2, 2, "l1", True),
        (2, 2, 2, 2, "dst", True),
        (2, 4, 2, 3, "l1", True),
    ]

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for rt, ct, kt, kb, mode, bias in cases:
            got, want = run(device, rt, ct, kt, kb, mode, bias)
            e = err(got, want)
            ok = e <= args.tol
            logger.info(
                f"A({rt}x{kt}t) @ B({ct}x{kt}t).T  k_blocks={kb} mode={mode:3s} "
                f"bias={int(bias)}  err={e:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append((rt, ct, kt, kb, mode, bias))
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
