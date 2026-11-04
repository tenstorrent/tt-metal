# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Source safety of NoC write flushes.

A sender stores a tag into an L1 word, issues a 4 B NoC write from that word to a receiver slot, flushes,
and immediately overwrites the word with a different tag. If the flush returns before the NoC has read
the word, the receiver gets the overwritten tag ("stale"). This is the pattern of a multicast sender that
resets its own semaphore right after flushing.

noc_async_writes_flushed() only waits until the NIU has accepted the write (its L1 read granted), not
until the read is done, so on Wormhole the stale tag is observed. noc_async_writes_departed() waits until
the NIU has finished reading every write's source, so the stale tag must never be observed.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0

KERNEL_DIR = "tests/ttnn/unit_tests/operations/debug/kernels/noc_writes_departed"

# The race depends on NoC/L1 hardware timing, which the simulator does not model.
pytestmark = pytest.mark.skipif(
    os.getenv("TT_METAL_SIMULATOR") is not None, reason="NoC source-read timing is a hardware property"
)

FLUSH_MODES = {"flushed": 0, "departed": 1, "none": 2}
SLOT_BYTES = 16
RESULT_WORDS = 32
OK, STALE, PREVIOUS, OTHER, PASSES = range(5)
SAMPLES = 8


def run_flush_race(device, flush_mode, mcast, num_receivers=5, ring=4096, passes=20, data_bytes=0):
    """Runs the sender/receiver pair and returns the per-receiver result words."""
    grid = device.compute_with_storage_grid_size()
    if grid.x < num_receivers + 1:
        pytest.skip(f"needs {num_receivers + 1} cores in a row, grid is {grid.x}x{grid.y}")
    if not mcast:
        num_receivers = 1

    sender = ttnn.CoreCoord(0, 0)
    receivers = [ttnn.CoreCoord(x, 0) for x in range(1, num_receivers + 1)]
    sender_virt = device.worker_core_from_logical_core(sender)
    receivers_virt = [device.worker_core_from_logical_core(c) for c in receivers]
    if mcast:
        xs = [c.x for c in receivers_virt]
        if sorted(xs) != list(range(min(xs), min(xs) + num_receivers)):
            pytest.skip("receiver cores are not contiguous in NoC coordinates")
        # NOC1 multicast rectangles are specified from the high corner to the low corner.
        start, end = (max(xs), receivers_virt[0].y), (min(xs), receivers_virt[0].y)
    else:
        start = end = (receivers_virt[0].x, receivers_virt[0].y)

    # One L1 scratch buffer at the same address on every core: ring, results/cell/ack, 16 KB data.
    rows_per_core = ring + 64 + 1024
    assert data_bytes <= 1024 * SLOT_BYTES
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(sender, receivers[-1])])
    num_cores = num_receivers + 1
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [rows_per_core, 4], ttnn.ShardOrientation.ROW_MAJOR),
    )
    scratch = ttnn.from_torch(
        torch.zeros(num_cores * rows_per_core, 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    # generic_op needs an input and an output tensor; the scratch buffer is the output.
    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    base = scratch.buffer_address()

    config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    sender_args = ttnn.RuntimeArgs()
    sender_args[sender.x][sender.y] = [base, start[0], start[1], end[0], end[1]]
    receiver_args = ttnn.RuntimeArgs()
    for c in receivers:
        receiver_args[c.x][c.y] = [base, sender_virt.x, sender_virt.y]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)]),
            compile_time_args=[ring, passes, FLUSH_MODES[flush_mode], num_receivers, int(mcast), data_bytes],
            runtime_args=sender_args,
            config=config,
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(receivers[0], receivers[-1])]),
            compile_time_args=[ring, passes],
            runtime_args=receiver_args,
            config=config,
        ),
    ]
    output = ttnn.generic_op([dummy, scratch], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[]))

    words = (ttnn.to_torch(output).to(torch.int64) & 0xFFFFFFFF).reshape(num_cores, rows_per_core * 4)
    results = []
    for k in range(1, num_cores):
        r = words[k][ring * 4 : ring * 4 + RESULT_WORDS].tolist()
        assert r[PASSES] == passes, f"receiver {k} finished {r[PASSES]} of {passes} passes"
        samples = [
            (r[SAMPLES + 2 * s] >> 16, r[SAMPLES + 2 * s] & 0xFFFF, hex(r[SAMPLES + 2 * s + 1]))
            for s in range(4)
            if r[SAMPLES + 2 * s + 1]
        ]
        results.append({"ok": r[OK], "stale": r[STALE], "previous": r[PREVIOUS], "other": r[OTHER], "samples": samples})
    return results, ring * passes


@pytest.mark.parametrize("mcast", [False, True], ids=["unicast", "multicast"])
@pytest.mark.parametrize("data_bytes", [0, 10880], ids=["flag_only", "with_payload"])
def test_noc_writes_departed_protects_source(device, mcast, data_bytes):
    # Control: with no wait at all, the detector must see the overwritten value.
    control, _ = run_flush_race(device, "none", mcast, passes=1, data_bytes=data_bytes)
    assert all(r["stale"] > 0 for r in control), f"stale-source detector did not fire: {control}"

    results, writes = run_flush_race(device, "departed", mcast, data_bytes=data_bytes)
    for k, r in enumerate(results):
        assert r["ok"] == writes and r["stale"] == r["previous"] == r["other"] == 0, f"receiver {k + 1}: {r}"


@pytest.mark.skipif(not is_wormhole_b0(), reason="the race reproduces on Wormhole; Blackhole's window is too short")
def test_noc_writes_flushed_sends_stale_source_on_wormhole(device):
    # Same pattern as the multicast sender that hung test_sharded_matmul_2d: a 10880 B linked payload
    # multicast, then a flag multicast from the sender's own semaphore, flush, and reset of that semaphore.
    results, writes = run_flush_race(device, "flushed", mcast=True, passes=50, data_bytes=10880)
    stale = sum(r["stale"] for r in results)
    assert stale > 0, (
        f"noc_async_writes_flushed() never let the NoC read the overwritten source in {writes} writes; "
        f"if this starts passing, the hardware or the flush changed: {results}"
    )
