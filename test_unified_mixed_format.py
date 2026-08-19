# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""An SFPU tree whose two leaves live in circular buffers of DIFFERENT data formats.

in0 is bfloat16, in1 is float32, and one expression reads both. Every other test in this
suite is uniformly bfloat16, which is exactly why this one exists: copy_tile does not
carry a data format, and copy_tile_to_dst_init_short explicitly "does not reconfigure the
unpacker data types". Without a per-leaf reconfig the second leaf is unpacked using the
first leaf's format and the result is quietly wrong -- no hang, no assert, just numbers.

Sabotage that proves it: forcing the leaf's `reconfigure` to false makes this test fail
while every uniformly-bfloat16 test keeps passing.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_mixed_format.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/binary.cpp"
CB_IN0, CB_IN1, CB_OUT = 0, 1, 16
TILE = 32


def run(device, num_blocks=2, tiles_per_block=2, rhs_dtype=ttnn.float32, seed=0):
    num_tiles = num_blocks * tiles_per_block
    shape = [1, num_tiles, TILE, TILE]

    torch.manual_seed(seed)
    a = (0.5 + 1.5 * torch.rand(shape)).to(torch.bfloat16)
    b = 0.5 + 1.5 * torch.rand(shape)  # kept in fp32

    torch_rhs = torch.float32 if rhs_dtype == ttnn.float32 else torch.bfloat16
    b = b.to(torch_rhs)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=rhs_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()
    ct_args = [num_blocks, tiles_per_block]
    for t in (ta, tb, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [ta.buffer_address(), tb.buffer_address(), tout.buffer_address()]

    # The point of the test: in1's circular buffer carries a different data format, and
    # therefore a different page size, from in0's.
    cbs = [
        make_cb(CB_IN0, core_ranges, dtype=ttnn.bfloat16, num_pages=2 * tiles_per_block),
        make_cb(CB_IN1, core_ranges, dtype=rhs_dtype, num_pages=2 * tiles_per_block),
        make_cb(CB_OUT, core_ranges, dtype=ttnn.bfloat16, num_pages=2 * tiles_per_block),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        defines=[("BN_MUL", "1")],
    )

    logger.info(f"running mixed-format binary: in0=bfloat16 in1={rhs_dtype} tiles={num_tiles}")
    out = ttnn.generic_op([ta, tb, tout], program)

    got = ttnn.to_torch(out).to(torch.float32)
    want = a.to(torch.float32) * b.to(torch.float32)
    return got, want


def max_rel_err(got, want):
    return ((got - want).abs() / want.abs().clamp(min=1e-6)).max().item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rel-err", type=float, default=0.02)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        # The bfloat16 row is the control: it must pass whether or not the reconfig is
        # there, which is what makes the float32 row the one carrying the information.
        for label, dt in (("in1 float32 (mixed)", ttnn.float32), ("in1 bfloat16 (control)", ttnn.bfloat16)):
            got, want = run(device, rhs_dtype=dt)
            rel = max_rel_err(got, want)
            ok = rel <= args.rel_err
            logger.info(f"{label:24s} max rel err = {rel:.5f}   {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(label)
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
