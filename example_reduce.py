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
from unified_harness import dfb, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/example_reduce.cpp"
TILE = 32
NUM_CORES = 4
IN_HT, IN_WT = 4, 2


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

        tensors = {"in": tx, "out": tout}
        spec = unified_program_spec(
            kernel_source=KERNEL,
            nodes=core_ranges,
            dfbs=[
                dfb("in", IN_HT * IN_WT),
                dfb("scaler", 1),
                dfb("partial", IN_WT),
                dfb("gathered", NUM_CORES * IN_WT),
                dfb("out", IN_WT),
            ],
            tensors=tensors,
            name="example_reduce",
        )
        run_unified_spec(device, spec, tensors)
        out = tout
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
