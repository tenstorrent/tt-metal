# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multicast broadcast across a row of cores, driven by host-allocated semaphores.

Core (0,0) reads one block and multicasts it to the whole row; every core then
computes exp(x) over its copy and writes its own output slice. All slices must
match, so a mis-addressed semaphore or a skipped handshake shows up as garbage in
one of them rather than as a silent pass.

    python test_unified_mcast.py --row 2 --tiles 2
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb_input, dfb_output, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/mcast_bcast.cpp"
TILE = 32


def run(device, row=2, tiles=2, dm_thread=0, barrier=False, seed=0):
    torch.manual_seed(seed)
    a = (torch.rand([1, 1, tiles * TILE, TILE]) - 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    # One output block per core, stacked along the tile-row axis.
    tout = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, row * tiles * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(row - 1, 0))])
    cores = [ttnn.CoreCoord(x, 0) for x in range(row)]

    named_ct_args = [("tiles_per_block", tiles)]
    defines = [
        ("MC_ROW_W", str(row)),
        # NOC 0: ttsim cannot express an ascending virtual mcast rect on NOC 1.
        ("MC_DM_THREAD", str(dm_thread)),
    ] + ([("MC_BARRIER", "1")] if barrier else [])

    logger.info(f"running unified mcast: row={row} tiles={tiles} dm_thread={dm_thread} " f"barrier={barrier}")

    # The broadcast runs on MC_DM_THREAD and the store on thread 0, which is what the two
    # roles below have to say. The six handshake semaphores are reserved by the harness;
    # tt/unified/api.h checks their base and contiguity against the ids the host assigned.
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb_input("in", thread=dm_thread, num_pages=tiles),
            dfb_output("out", thread=0, num_pages=tiles),
        ],
        tensors={"in": ta, "out": tout},
        named_compile_time_args=named_ct_args,
        runtime_arg_names=["out_block"],
        defines=defines,
        name="mcast_bcast",
    )
    run_unified_spec(
        device,
        spec,
        {"in": ta, "out": tout},
        runtime_args={"out_block": {c: i for i, c in enumerate(cores)}},
    )
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)
    want = torch.exp(a.to(torch.float32)).repeat(1, 1, row, 1)
    return got, want


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=int, default=2, help="cores in the multicast row")
    p.add_argument("--tiles", type=int, default=2)
    p.add_argument("--dm-thread", type=int, default=0, choices=[0, 1], help="which DM thread broadcasts")
    p.add_argument("--barrier", action="store_true", help="synchronize_cores() twice mid-kernel")
    p.add_argument("--pcc", type=float, default=0.99)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        got, want = run(device, args.row, args.tiles, args.dm_thread, args.barrier)
    finally:
        ttnn.close_device(device)

    measured = pcc(got, want)
    logger.info(f"PCC = {measured:.6f} (threshold {args.pcc})")
    # Every core's slice must match core 0's, or the broadcast didn't reach it.
    rows = got.shape[2] // args.row
    slices = [got[0, 0, i * rows : (i + 1) * rows, :] for i in range(args.row)]
    spread = max((s - slices[0]).abs().max().item() for s in slices)
    logger.info(f"max |slice_i - slice_0| = {spread:.6f}  (0 means all cores got the same block)")
    if measured < args.pcc:
        logger.error("FAIL")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
