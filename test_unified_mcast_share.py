# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Does sharing a handshake pair break when the two collectives have the SAME rectangle?

Hazard 13b is proven for DIFFERENT rectangles. This asks whether rectangle equality is
what makes sharing safe, or whether two transactions on one semaphore is unsafe on its own.

unified_kernels/mcast_share.cpp broadcasts one operand TWICE over the same 8x8 rectangle
into two buffers, on the same pair. Same sender, same receivers, same extent. Both outputs
must equal the input on every core, so a protocol failure is a hang or a wrong buffer --
and there is no arithmetic to confuse it with.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_mcast_share.py
"""

import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, unified_program_spec

KERNEL = "unified_kernels/mcast_share.cpp"
TILE = 32


def run(device, grid_h=8, grid_w=8, tiles=2, rounds=8, share_pair=False, skew=0, seed=0, dynamic_noc=False):
    cores_n = grid_h * grid_w
    torch.manual_seed(seed)
    src = (torch.rand([1, rounds * tiles, TILE, TILE]) - 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

    tin = to_dev(src)
    # One slice per core, NaN-filled so a core that never wrote is unmistakable.
    blank = torch.full([1, cores_n * rounds * tiles, TILE, TILE], float("nan")).to(torch.bfloat16)
    tout0, tout1 = to_dev(blank), to_dev(blank)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))])
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_h) for x in range(grid_w)]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb("a", tiles),
            dfb("b", tiles),
            dfb("out0", tiles),
            dfb("out1", tiles),
        ],
        named_compile_time_args=[
            ("tiles", tiles),
            ("rounds", rounds),
            ("grid_h", grid_h),
            ("grid_w", grid_w),
        ],
        tensors={"in": tin, "out0": tout0, "out1": tout1},
        defines=([("MS_SHARE_PAIR", "1")] if share_pair else []) + ([("MS_SKEW", str(skew))] if skew else []),
        dynamic_noc=dynamic_noc,
    )

    run_unified_spec(device, spec, {"in": tin, "out0": tout0, "out1": tout1})
    got0 = ttnn.to_torch(tout0).to(torch.float32)
    got1 = ttnn.to_torch(tout1).to(torch.float32)
    # Every core writes all `rounds` blocks in order, so the expectation is the whole
    # source repeated once per core.
    want = src.to(torch.float32).repeat(1, cores_n, 1, 1)
    return got0, got1, want


def main():
    # The second axis is NOC mode. unified_explicit_noc_spec.md 13.1: under
    # DM_DYNAMIC_NOC the flush predicates sum BOTH RISCs' counters, so the handshake's
    # noc_async_writes_flushed() now waits on the other thread's writes too. That should
    # be over-waiting -- slow, not wrong -- but the handshake is the one protocol where
    # "should be" has not been checked.
    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for dyn in (False, True):
            mode = "DYNAMIC " if dyn else "dedicated"
            for label, share, sk in (
                ("distinct pairs, no skew ", False, 0),
                ("SHARED pair, no skew    ", True, 0),
                ("distinct pairs, SKEWED  ", False, 5000),
                ("SHARED pair, SKEWED     ", True, 5000),
            ):
                got0, got1, want = run(device, share_pair=share, skew=sk, dynamic_noc=dyn)
                e0 = (got0 - want).abs().max().item()
                e1 = (got1 - want).abs().max().item()
                # Exact: a broadcast copies bits, and copy() is a pack of what was unpacked.
                ok = e0 == 0.0 and e1 == 0.0
                logger.info(
                    f"  [{mode}] 8x8 rect x8 rounds, {label}: "
                    f"max|err| buf0={e0:.6f} buf1={e1:.6f}   {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append(f"{mode} {label}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
