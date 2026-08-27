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
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/matmul.cpp"

TILE = 32


def run(device, rt, ct, kt, k_blocks=1, relu=None, mode="dst", seed=0, fidelity=None, bias_epilogue=False):
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

    dfbs = [
        dfb("in0", rt * kt),
        dfb("in1", kt * ct),
        dfb("bias", ct),
        dfb("out", rt * ct),
        dfb("acc", rt * ct),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors={"in0": ta, "in1": tb, "out": tout, "bias": tbias},
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
            + ([("MM_BIAS_EPILOGUE", "1")] if bias_epilogue else [])
        ),
        # So a sweep can pin fidelity to match whatever it compares against.
        **(fidelity or {}),
    )

    logger.info(f"running biased matmul: rt={rt} ct={ct} kt={kt} k_blocks={k_blocks} mode={mode} relu={relu}")
    # Output last: generic_op hands back the final tensor in the list. That order is
    # independent of the accessor-arg order above, where bias must stay last so a
    # build without MM_BIAS sees the layout the other matmul tests use.
    run_unified_spec(device, spec, {"in0": ta, "in1": tb, "out": tout, "bias": tbias})
    out = tout

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
    failed = []
    try:
        got, want = run(device, args.rt, args.ct, args.kt, args.k_blocks, args.relu, args.mode)

        # The SAME bias, written in the epilogue instead of on the fusion. Both spellings
        # mean "added once to the finished total", so they must agree with each other and
        # with the reference -- and the epilogue one is the spelling whose timing is
        # stated rather than special-cased. It is checked EXACTLY against the fusion
        # spelling, because it lowers to the same instructions and anything else is a
        # difference worth seeing.
        #
        # k_blocks > 1 is what gives this teeth twice over: a bias applied per block, and
        # an epilogue operand DROPPED (which is what evaluating the lambda fixed -- it
        # compiled and produced an unbiased matmul, 0.49 max error on a +-0.5 bias).
        epi = None
        if args.mode != "single":
            epi, _ = run(device, args.rt, args.ct, args.kt, args.k_blocks, args.relu, args.mode, bias_epilogue=True)
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    err = (got - want).abs().max().item()
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    # A bias added k_blocks times is a per-column offset: correlation barely
    # notices it, so the absolute error is the check that actually bites.
    logger.info(f"max |got - want| = {err:.4f} (threshold {args.atol})")
    if measured < args.pcc or err > args.atol:
        failed.append("fusion-bias")

    if epi is not None:
        same = (epi - got).abs().max().item()
        epi_err = (epi - want).abs().max().item()
        logger.info(f"bias in the EPILOGUE: max |epi - fusion| = {same:.6f}, max |epi - want| = {epi_err:.4f}")
        if same != 0.0 or epi_err > args.atol:
            failed.append("epilogue-bias")

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
