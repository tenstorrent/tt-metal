# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runs unified_kernels/example_reduce.cpp: a column sum reduced by a tree of 4 cores.

Each core sums its own 4x2-tile block down to one row, writes that row straight into the
root core's L1, and the root sums the four gathered rows again. Every core runs the same
source; which side of the gather it takes it decides from its own coordinate.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python example_reduce.py
"""

import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, unified_program

KERNEL = "unified_kernels/example_reduce.cpp"
TILE = 32
NUM_CORES = 4
IN_HT, IN_WT = 4, 2

CB_IN, CB_SCALER, CB_PARTIAL, CB_GATHERED, CB_OUT = 0, 1, 2, 3, 16


def to_device(device, t):
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    torch.manual_seed(0)
    rows = NUM_CORES * IN_HT * TILE
    x = ((torch.rand([rows, IN_WT * TILE]) - 0.5) / rows**0.5).to(torch.bfloat16)

    device = ttnn.open_device(device_id=0)
    try:
        tx = to_device(device, x)
        tout = to_device(device, torch.full([TILE, IN_WT * TILE], float("nan")))

        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, NUM_CORES - 1))])
        cores = [ttnn.CoreCoord(0, y) for y in range(NUM_CORES)]

        compile_time_args = []
        for t in (tx, tout):
            compile_time_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        program = unified_program(
            kernel_source=KERNEL,
            core_ranges=core_ranges,
            cores=cores,
            cbs=[
                make_cb(CB_IN, core_ranges, num_pages=IN_HT * IN_WT),
                make_cb(CB_SCALER, core_ranges, num_pages=1),
                make_cb(CB_PARTIAL, core_ranges, num_pages=IN_WT),
                make_cb(CB_GATHERED, core_ranges, num_pages=NUM_CORES * IN_WT),
                make_cb(CB_OUT, core_ranges, num_pages=IN_WT),
            ],
            compile_time_args=compile_time_args,
            runtime_args=[t.buffer_address() for t in (tx, tout)],
        )

        out = ttnn.generic_op([tx, tout], program)
        got = ttnn.to_torch(out).to(torch.float32)[0, 0]
    finally:
        ttnn.close_device(device)

    want = x.to(torch.float32).sum(dim=0)
    error = ((got[0] - want).norm() / want.norm()).item()
    logger.info(f"column sum of {rows} rows over {NUM_CORES} cores: relative error = {error:.5f}")
    if error > 0.02:
        logger.error("FAIL")
        return 1
    logger.info("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
