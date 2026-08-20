# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the unified unary ops on device: recip, sqrt, rsqrt, and a two-op chain.

One kernel source (unified_kernels/unary.cpp) compiled for all five baby RISC-V
threads via three KernelDescriptors. See unified_harness.py.

PCC is reported but is NOT the check. The gate is max RELATIVE error, elementwise.
That is not a precaution: deleting the sqrt from the chain below leaves PCC at
0.9958 -- above the 0.99 threshold, so a PCC-only test PASSES the sabotage -- while
the relative error goes to 0.414.

The chain case is also checked against the device's own rsqrt, since
recip(sqrt(x)) == rsqrt(x) by identity. That is a torch-independent consistency
check on how a two-op SFPU chain composes in DST, not a stronger test than the
comparison against torch: a dropped link fails both.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_unary.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/unary.cpp"

CB_IN, CB_OUT = 0, 16

# name -> (kernel define, reference)
OPS = {
    "recip": (None, lambda x: 1.0 / x),
    "sqrt": ("UN_SQRT", torch.sqrt),
    "rsqrt": ("UN_RSQRT", torch.rsqrt),
    "exp": ("UN_EXP", torch.exp),
    "chain": ("UN_CHAIN", torch.rsqrt),  # recip(sqrt(x)) == rsqrt(x)
}


def run(device, op, num_blocks=1, tiles_per_block=1, seed=0, fidelity=None):
    num_tiles = num_blocks * tiles_per_block
    shape = [1, num_tiles, 32, 32]

    torch.manual_seed(seed)
    # Strictly positive and away from zero: sqrt and rsqrt are undefined at or
    # below it, and recip is ill-conditioned near it.
    a = (0.5 + 1.5 * torch.rand(shape)).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()

    ct_args = [num_blocks, tiles_per_block]
    for t in (ta, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    rt_args = [ta.buffer_address(), tout.buffer_address()]

    cbs = [
        make_cb(CB_IN, core_ranges, num_pages=2 * tiles_per_block),
        make_cb(CB_OUT, core_ranges, num_pages=2 * tiles_per_block),
    ]

    define, reference = OPS[op]
    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        defines=[(define, "1")] if define else None,
        **(fidelity or {}),
    )

    logger.info(f"running unified unary: op={op} num_blocks={num_blocks} tiles_per_block={tiles_per_block}")
    out = ttnn.generic_op([ta, tout], program)

    got = ttnn.to_torch(out).to(torch.float32)
    want = reference(a.to(torch.float32))
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def max_rel_err(got, want):
    return ((got - want).abs() / want.abs().clamp(min=1e-6)).max().item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=list(OPS) + ["all"], default="all")
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--tiles-per-block", type=int, default=2)
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument("--rel-err", type=float, default=0.02, help="max elementwise relative error")
    args = p.parse_args(argv)

    ops = list(OPS) if args.op == "all" else [args.op]
    results = {}

    device = ttnn.open_device(device_id=0)
    try:
        for op in ops:
            results[op] = run(device, op, args.num_blocks, args.tiles_per_block)
    finally:
        ttnn.close_device(device)

    failed = []
    for op in ops:
        got, want = results[op]
        measured, rel = pcc(got, want), max_rel_err(got, want)
        ok = measured >= args.pcc and rel <= args.rel_err
        logger.info(f"{op:6s} PCC = {measured:.6f}   max rel err = {rel:.5f}   {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(op)

    # recip(sqrt(x)) against the device's own rsqrt, not against torch: this is
    # the only check that would notice a dropped link in the chain.
    if "chain" in results and "rsqrt" in results:
        spread = (results["chain"][0] - results["rsqrt"][0]).abs().max().item()
        scale = results["rsqrt"][0].abs().max().item()
        logger.info(f"chain vs rsqrt: max |diff| = {spread:.5f} (values up to {scale:.3f})")
        if spread > args.rel_err * scale:
            logger.error("chain and rsqrt disagree beyond tolerance")
            failed.append("chain-vs-rsqrt")

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
