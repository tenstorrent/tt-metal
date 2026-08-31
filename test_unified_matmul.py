# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a unified matmul on device: C = A @ B for one output subblock.

Exercises the FPU fusion path (unified_kernels/matmul.cpp), where matmul_block
owns the whole DST register file and the k-loop.

    export TT_METAL_HOME=$PWD
    export TT_METAL_SIMULATOR_HOME=$HOME/sim TT_METAL_SIMULATOR=$HOME/sim/libttsim.so
    export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
    source python_env/bin/activate
    python test_unified_matmul.py --rt 1 --ct 1 --kt 2
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/matmul.cpp"
TILE = 32


def run(device, rt, ct, kt, k_blocks=1, relu=None, mode="dst", seed=0, fidelity=None):
    """mode: "dst" or "l1" accumulate through a buffer; "single" stores straight out."""
    """relu: None, "epilogue" (finish only), or "per_step"."""
    torch.manual_seed(seed)
    # The k dimension is split into k_blocks blocks of kt tiles each. The reader
    # streams one block per iteration, so A is laid out block-major: block k is
    # a contiguous rt x kt tile region, and likewise for B.
    ktot = kt * k_blocks
    a_blocks = [(torch.rand([1, 1, rt * TILE, kt * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]
    b_blocks = [(torch.rand([1, 1, kt * TILE, ct * TILE]) - 0.5).to(torch.bfloat16) for _ in range(k_blocks)]
    # Concatenate along the tile-row axis so each block occupies consecutive pages.
    a = torch.cat(a_blocks, dim=2)  # (k_blocks*rt) x kt  tiles
    b = torch.cat(b_blocks, dim=2)  # (k_blocks*kt) x ct  tiles

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rt * TILE, ct * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges, cores = single_core()

    # Each operand DFB must hold its whole block: the k-loop indexes tiles inside
    # it, so partial residency is not an option here.
    dfbs = [
        dfb("in0", rt * kt),
        dfb("in1", kt * ct),
        dfb("out", rt * ct),
        dfb("acc", rt * ct),
        # Declared even without MM_BIAS: the kernel declares its Storage unconditionally.
        dfb("bias", ct),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors={"in0": ta, "in1": tb, "out": tout},
        defines=(
            [("MM_RT_DIM", str(rt)), ("MM_CT_DIM", str(ct)), ("MM_KT_DIM", str(kt)), ("MM_K_BLOCKS", str(k_blocks))]
            + ([("MM_ACC_L1", "1")] if mode == "l1" else [])
            + ([("MM_SINGLE_SHOT", "1")] if mode == "single" else [])
            + ([("MM_RELU_EPILOGUE", "1")] if relu == "epilogue" else [])
            + ([("MM_RELU_PER_STEP", "1")] if relu == "per_step" else [])
            + ([("MM_RELU_BOTH", "1")] if relu == "both" else [])
        ),
        **(fidelity or {}),
    )

    logger.info(f"running unified matmul: rt={rt} ct={ct} kt={kt} k_blocks={k_blocks} mode={mode} relu={relu}")
    run_unified_spec(device, spec, {"in0": ta, "in1": tb, "out": tout})
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)
    # Sum of per-block products, with the chains applied where the hardware
    # applies them. See Strategy<FPUFusion>::run: a per-step chain sees this
    # block's contribution alone in L1 mode (the packer does the summing) but
    # the running total in Dst mode (the reload precedes the matmul).
    want = torch.zeros([1, 1, rt * TILE, ct * TILE], dtype=torch.float32)
    for ab, bb in zip(a_blocks, b_blocks):
        p = ab.to(torch.float32) @ bb.to(torch.float32)
        if relu in ("per_step", "both"):
            want = want + torch.relu(p) if mode == "l1" else torch.relu(want + p)
        else:
            want = want + p
    if relu == "epilogue":
        want = torch.relu(want)
    elif relu == "both":
        want = torch.exp(want)
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rt", type=int, default=1, help="output rows in tiles")
    p.add_argument("--ct", type=int, default=1, help="output cols in tiles")
    p.add_argument("--kt", type=int, default=2, help="inner dim in tiles")
    p.add_argument("--k-blocks", type=int, default=1, help="k-blocks to accumulate over")
    p.add_argument(
        "--relu",
        choices=["epilogue", "per_step", "both"],
        default=None,
        help="epilogue = relu once on the finished accumulator; per_step = relu every k-block",
    )
    p.add_argument("--mode", choices=["dst", "l1"], default="dst", help="how the running total is carried")
    p.add_argument("--pcc", type=float, default=0.99)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(device, args.rt, args.ct, args.kt, args.k_blocks, args.relu, args.mode)
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    logger.info(f"got [:4]  = {got.flatten()[:4].tolist()}")
    logger.info(f"want[:4]  = {want.flatten()[:4].tolist()}")
    if measured < args.pcc:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
