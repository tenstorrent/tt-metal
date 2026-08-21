# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fused bias on a matmul: out = A @ B + bias, bias broadcast down the rows.

This is the contract for `matmul<Geom>(a, b).bias(bias_storage)`. The interesting
part is the interaction with Accumulator: bias is added ONCE, to the finished
total, not once per k-block. A per-block implementation returns A@B + k_blocks*bias
and --k-blocks 3 is what catches it.

Ordering is the other thing pinned here: a trailing relu must see the biased value,
i.e. relu(A@B + bias) and not relu(A@B) + bias.

    python test_unified_matmul_bias.py --k-blocks 3 --mode l1 --relu epilogue
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/matmul.cpp"

CB_IN0, CB_IN1, CB_BIAS, CB_OUT, CB_ACC = 0, 1, 2, 16, 24
TILE = 32


def run(device, rt, ct, kt, k_blocks=1, relu=None, mode="dst", seed=0, fidelity=None):
    torch.manual_seed(seed)
    a_blocks = [(torch.rand([1, 1, rt * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]
    b_blocks = [(torch.rand([1, 1, kt * TILE, ct * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]
    a = torch.cat(a_blocks, dim=2)
    b = torch.cat(b_blocks, dim=2)

    # One row of bias per output column, held as ct tiles -- REPLICATED down all 32 rows
    # of each tile. The folded path adds the bias with an FPU dest-reuse add, which is
    # elementwise and does no broadcasting, so it needs every row to carry the value. The
    # two-pass path reads row 0 and broadcasts it in hardware, and row 0 is unchanged, so
    # replication is correct for both and the two are directly comparable.
    #
    # This does cost a diagnostic: rows 1..31 used to be zeroed so that a broadcast that
    # failed showed up as zeros rather than noise. With every row equal, a broken row
    # broadcast is invisible -- which is fine for the folded path, since it does not
    # broadcast at all, but it means the two-pass path's broadcast is no longer covered.
    # Deliberately larger than a typical A@B entry, so applying it the wrong number
    # of times -- or before a relu instead of after -- is unmistakable.
    bias_row = ((torch.rand([ct * TILE]) - 0.5) * 4.0).to(torch.bfloat16)
    bias = torch.zeros([1, 1, TILE, ct * TILE], dtype=torch.bfloat16)
    bias[0, 0, :, :] = bias_row

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tbias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rt * TILE, ct * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges, cores = single_core()

    # bias args go last, so a build without MM_BIAS keeps the existing layout.
    ct_args = []
    for t in (ta, tb, tout, tbias):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [ta.buffer_address(), tb.buffer_address(), tout.buffer_address(), tbias.buffer_address()]

    cbs = [
        make_cb(CB_IN0, core_ranges, num_pages=rt * kt),
        make_cb(CB_IN1, core_ranges, num_pages=kt * ct),
        # Pushed once and never popped, like the reduce scaler: every finishing
        # block re-reads the same ct tiles.
        make_cb(CB_BIAS, core_ranges, num_pages=ct),
        make_cb(CB_OUT, core_ranges, num_pages=rt * ct),
        make_cb(CB_ACC, core_ranges, num_pages=rt * ct),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        defines=(
            [
                ("MM_RT_DIM", str(rt)),
                ("MM_CT_DIM", str(ct)),
                ("MM_KT_DIM", str(kt)),
                ("MM_K_BLOCKS", str(k_blocks)),
                ("MM_BIAS", "1"),
            ]
            + ([("MM_ACC_L1", "1")] if mode == "l1" else [])
            + ([("MM_SINGLE_SHOT", "1")] if mode == "single" else [])
            + ([("MM_RELU_EPILOGUE", "1")] if relu == "epilogue" else [])
        ),
        # So a sweep can pin fidelity to match whatever it compares against.
        **(fidelity or {}),
    )

    logger.info(f"running biased matmul: rt={rt} ct={ct} kt={kt} k_blocks={k_blocks} mode={mode} relu={relu}")
    # Output last: generic_op hands back the final tensor in the list. That order is
    # independent of the accessor-arg order above, where bias must stay last so a
    # build without MM_BIAS sees the layout the other matmul tests use.
    out = ttnn.generic_op([ta, tb, tbias, tout], program)

    got = ttnn.to_torch(out).to(torch.float32)
    want = torch.zeros([1, 1, rt * TILE, ct * TILE], dtype=torch.float32)
    for ab, bb in zip(a_blocks, b_blocks):
        want = want + ab.to(torch.float32) @ bb.to(torch.float32)
    want = want + bias_row.to(torch.float32)  # ONCE, on the finished total
    if relu == "epilogue":
        want = torch.relu(want)  # after the bias, not before
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rt", type=int, default=2)
    p.add_argument("--ct", type=int, default=2)
    p.add_argument("--kt", type=int, default=2)
    p.add_argument("--k-blocks", type=int, default=1, help=">1 is what catches bias applied per block")
    p.add_argument("--relu", choices=["epilogue"], default=None)
    p.add_argument("--mode", choices=["dst", "l1", "single"], default="dst")
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument("--atol", type=float, default=0.2, help="PCC alone tolerates a systematic offset")
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(device, args.rt, args.ct, args.kt, args.k_blocks, args.relu, args.mode)
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    err = (got - want).abs().max().item()
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    # A bias added k_blocks times is a per-column offset: correlation barely
    # notices it, so the absolute error is the check that actually bites.
    logger.info(f"max |got - want| = {err:.4f} (threshold {args.atol})")
    if measured < args.pcc or err > args.atol:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
