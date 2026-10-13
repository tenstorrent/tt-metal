# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A data-movement kernel must not return with NoC responses still in flight.

The next kernel on the same RISC snapshots the NIU counters in noc_local_state_init(), before its go message and,
with fast dispatch, possibly before the previous kernel has been reported done (its launch message is preloaded). Its
barriers compare the counters with that snapshot for equality, so a response to the previous kernel that lands after
the snapshot leaves the counter ahead of the software count and the barrier never returns.

The producer program returns with non-posted semaphore increments still in flight to a far core while another core
floods the path to it. The observer program, enqueued directly behind, checks whether any atomic response arrived
after its snapshot and evaluates the atomic barrier predicate with a cap instead of the API's unbounded loop. The
observation waits are capped; an outer process timeout (GNU timeout --kill-after=10s) bounds execution, for example:

    timeout --signal=TERM --kill-after=10s 900s pytest tests/ttnn/unit_tests/operations/debug/test_kernel_exit_noc_drain.py
"""

import os

import pytest
import torch
import ttnn

KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "kernel_exit_noc_drain")

# Offsets inside the 16 KiB CB page that both programs place at the same L1 address on cores A, B and F.
PAGE_SIZE = 16384
FLOOD_OFFSET = 4096  # core F writes 4 KiB here on core B
NOTE_OFFSET = 8192  # producer -> observer notes on core A
SEMAPHORE_OFFSET = 12544  # the word on core B that the producer and the observer increment
RESULT_OFFSET = 14336  # observer result block on core A, 2 KiB

ATOMICS = 200  # semaphore increments the producer leaves in flight
FLOOD_WRITES = 100000  # 4 KiB writes of the flooding core
WAIT_CAP = 300000  # polls the producer waits for the observer's launch message to be preloaded
OBSERVER_POLLS = 20000
OWN_ATOMICS = 2
CAP = 2000000
NEVER = 0xFFFFFFFF


def _cores(*cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in cores})


def _kernel(name, cores, runtime_args, compile_time_args=None):
    return ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNELS, name),
        core_ranges=_cores(*cores),
        compile_time_args=compile_time_args or [],
        runtime_args=runtime_args,
        config=ttnn.WriterConfigDescriptor(),
    )


def _programs(device, producer, target, flooder, mode, result):
    cores = _cores(producer, target, flooder)
    cbs = [
        ttnn.CBDescriptor(
            total_size=PAGE_SIZE,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE_SIZE)
            ],
        )
    ]
    t = device.worker_core_from_logical_core(target)
    producer_args, observer_args, flood_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    producer_args[producer.x][producer.y] = [t.x, t.y, ATOMICS, mode, WAIT_CAP, NOTE_OFFSET, SEMAPHORE_OFFSET]
    observer_args[producer.x][producer.y] = [
        OBSERVER_POLLS,
        result.buffer_address(),
        RESULT_OFFSET,
        NOTE_OFFSET,
        OWN_ATOMICS,
        CAP,
        t.x,
        t.y,
        SEMAPHORE_OFFSET,
    ]
    flood_args[flooder.x][flooder.y] = [t.x, t.y, FLOOD_WRITES, FLOOD_OFFSET, CAP]
    first = ttnn.ProgramDescriptor(
        kernels=[
            _kernel("leaky_producer.cpp", [producer], producer_args),
            _kernel("idle_target.cpp", [target], ttnn.RuntimeArgs()),
            _kernel("flood_neighbour.cpp", [flooder], flood_args),
        ],
        cbs=cbs,
        semaphores=[],
    )
    second = ttnn.ProgramDescriptor(
        kernels=[
            _kernel(
                "exit_observer.cpp",
                [producer],
                observer_args,
                ttnn.TensorAccessorArgs(result).get_compile_time_args(),
            ),
            _kernel("idle_target.cpp", [target, flooder], ttnn.RuntimeArgs()),
        ],
        cbs=cbs,
        semaphores=[],
    )
    return first, second


@pytest.mark.timeout(300)
@pytest.mark.skipif(bool(os.getenv("TT_METAL_SIMULATOR")), reason="needs hardware NoC timing")
@pytest.mark.skipif(
    bool(os.getenv("TT_METAL_SLOW_DISPATCH_MODE")), reason="needs fast dispatch, which preloads launch messages"
)
@pytest.mark.parametrize("mode", [0, 1], ids=["return_with_atomics_in_flight", "atomic_barrier_control"])
def test_kernel_exit_leaves_no_late_atomic_responses(device, mode):
    if device.arch() not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip("covers the tt-1xx data-movement firmware")
    iterations = int(os.getenv("TT_METAL_KERNEL_EXIT_DRAIN_ITERATIONS", "64"))
    grid = device.compute_with_storage_grid_size()
    producer, flooder = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0)
    target = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    unused_input = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    result = ttnn.from_torch(
        torch.zeros((1, 1, 1, 512), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    first, second = _programs(device, producer, target, flooder, mode, result)

    preloaded, failures = 0, []
    for iteration in range(iterations):
        ttnn.generic_op([unused_input, result], first)
        ttnn.generic_op([unused_input, result], second)  # nothing in between: the second launch is preloaded
        r = (ttnn.to_torch(result).reshape(-1).to(torch.int64) & 0xFFFFFFFF).tolist()
        assert r[0] == 0xE0170004 and r[11] == 0xD0E5 and r[1] == 0xE0170003, f"unreadable result {r[:12]}"
        assert r[3] == mode and r[4] == ATOMICS, f"producer notes do not match: {r[:12]}"
        if not r[2]:
            continue  # the observer was not preloaded while the producer ran: no hand-off to test
        preloaded += 1
        late = (r[7] - r[5]) & 0xFFFFFFFF
        if late or not r[8] or r[9] == NEVER:
            failures.append(
                f"iteration {iteration}: {late} atomic responses arrived after the next kernel's counter snapshot, "
                f"barrier predicate with nothing issued {'held' if r[8] else 'false'}, "
                f"barrier after own atomics {'never held' if r[9] == NEVER else 'held'}"
            )
    assert preloaded, "the observer's launch message was never preloaded, so the hand-off was not exercised"
    assert not failures, f"{len(failures)} of {preloaded} preloaded iterations failed; first: {failures[0]}"
