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
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/matmul.cpp"
CB_IN0, CB_IN1, CB_OUT, CB_ACC = 0, 1, 16, 24
TILE = 32


def run(device, rt, ct, kt, k_blocks=1, relu=None, mode="dst", seed=0):
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

    ct_args = []
    for t in (ta, tb, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [ta.buffer_address(), tb.buffer_address(), tout.buffer_address()]

    # Each operand CB must hold its whole block: the k-loop indexes tiles inside
    # it, so partial residency is not an option here.
    cbs = [
        make_cb(CB_IN0, core_ranges, num_pages=rt * kt),
        make_cb(CB_IN1, core_ranges, num_pages=kt * ct),
        make_cb(CB_OUT, core_ranges, num_pages=rt * ct),
        # The running total. A separate CB from CB_OUT: intermediates are pushed
        # here and re-consumed by the next k-block, so the DM writer must not see
        # them. Sized to exactly one block, matching push/pop granularity.
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
            [("MM_RT_DIM", str(rt)), ("MM_CT_DIM", str(ct)), ("MM_KT_DIM", str(kt)), ("MM_K_BLOCKS", str(k_blocks))]
            + ([("MM_ACC_L1", "1")] if mode == "l1" else [])
            + ([("MM_RELU_EPILOGUE", "1")] if relu == "epilogue" else [])
            + ([("MM_RELU_PER_STEP", "1")] if relu == "per_step" else [])
            + ([("MM_RELU_BOTH", "1")] if relu == "both" else [])
        ),
    )

    logger.info(f"running unified matmul: rt={rt} ct={ct} kt={kt} k_blocks={k_blocks} mode={mode} relu={relu}")
    out = ttnn.generic_op([ta, tb, tout], program)

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
