# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a unified kernel on device: out = exp(in0 + in1).

One kernel source (unified_kernels/eltwise_add_exp.cpp) compiled for all five
baby RISC-V threads via three KernelDescriptors. See unified_harness.py.

    export TT_METAL_HOME=$PWD
    export TT_METAL_SIMULATOR_HOME=$HOME/sim TT_METAL_SIMULATOR=$HOME/sim/libttsim.so
    export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
    source python_env/bin/activate
    python test_unified_add_exp.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/eltwise_add_exp.cpp"


def run(device, num_blocks=1, tiles_per_block=1, custom_load=False, seed=0):
    num_tiles = num_blocks * tiles_per_block
    shape = [1, num_tiles, 32, 32]

    torch.manual_seed(seed)
    # Keep inputs small so exp() stays in bfloat16's comfortable range.
    a = (torch.rand(shape) * 0.5).to(torch.bfloat16)
    b = (torch.rand(shape) * 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()

    # CT args: [num_blocks, tiles_per_block] then TensorAccessorArgs for in0, in1, out
    named_ct_args = [("num_blocks", num_blocks), ("tiles_per_block", tiles_per_block)]

    # Double-buffer the inputs; the output DFB must hold a whole block because the
    # SFPU strategy reserves the block, packs each tile, then pushes once.
    dfbs = [
        dfb("in0", 2 * tiles_per_block),
        dfb("in1", 2 * tiles_per_block),
        dfb("out", 2 * tiles_per_block),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"in0": ta, "in1": tb, "out": tout},
        defines=[("EA_CUSTOM_LOAD", "1")] if custom_load else None,
    )

    logger.info(
        f"running unified kernel: num_blocks={num_blocks} tiles_per_block={tiles_per_block} custom_load={custom_load}"
    )
    run_unified_spec(device, spec, {"in0": ta, "in1": tb, "out": tout})
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)
    want = torch.exp(a.to(torch.float32) + b.to(torch.float32))
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--tiles-per-block", type=int, default=1)
    p.add_argument("--custom-load", action="store_true", help="fill the input DFBs with a user-written routine")
    p.add_argument("--pcc", type=float, default=0.99)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(device, args.num_blocks, args.tiles_per_block, args.custom_load)
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    logger.info(f"got [0,0,0,:4]  = {got.flatten()[:4].tolist()}")
    logger.info(f"want[0,0,0,:4]  = {want.flatten()[:4].tolist()}")
    if measured < args.pcc:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
