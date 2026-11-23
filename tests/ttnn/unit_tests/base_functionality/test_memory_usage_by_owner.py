# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import threading

import ttnn

SHAPE = ttnn.Shape((1, 1, 32, 32))  # one bfloat16 tile: 2048 B


def _allocate(device, memory_config):
    return ttnn.allocate_tensor_on_device(SHAPE, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)


def _usage(device, owner, buffer_type):
    for entry in ttnn.get_memory_usage_by_owner(device):
        if entry.owner == owner and entry.buffer_type == buffer_type:
            return entry
    return None


def test_memory_usage_is_attributed_to_the_innermost_owner(device):
    l1_input = _allocate(device, ttnn.L1_MEMORY_CONFIG)
    # Compile outside any owner so the program cache's buffers are not attributed below.
    ttnn.neg(l1_input)
    unowned_before = _usage(device, "", ttnn.BufferType.L1)

    with ttnn.memory_owner("model_a"):
        a_l1 = _allocate(device, ttnn.L1_MEMORY_CONFIG)
        a_dram = _allocate(device, ttnn.DRAM_MEMORY_CONFIG)
        with ttnn.memory_owner("model_b"):
            b_l1 = [_allocate(device, ttnn.L1_MEMORY_CONFIG) for _ in range(3)]
        # Outputs of ops run inside the scope belong to the owner as well.
        a_out = ttnn.neg(l1_input)
        assert ttnn.current_memory_owner() == "model_a"
    assert ttnn.current_memory_owner() == ""
    unowned = _allocate(device, ttnn.L1_MEMORY_CONFIG)

    # A one-page interleaved buffer reserves one page in every bank.
    tile_bytes = 2048
    a = _usage(device, "model_a", ttnn.BufferType.L1)
    assert (a.num_buffers, a.total_bytes, a.bytes_per_bank) == (2, 2 * 2048, 2 * tile_bytes)
    a_dram_usage = _usage(device, "model_a", ttnn.BufferType.DRAM)
    assert (a_dram_usage.num_buffers, a_dram_usage.total_bytes) == (1, 2048)
    b = _usage(device, "model_b", ttnn.BufferType.L1)
    assert (b.num_buffers, b.total_bytes, b.bytes_per_bank) == (3, 3 * 2048, 3 * tile_bytes)
    assert _usage(device, "model_b", ttnn.BufferType.DRAM) is None
    unowned_after = _usage(device, "", ttnn.BufferType.L1)
    assert unowned_after.num_buffers == unowned_before.num_buffers + 1

    # Freed buffers stop counting.
    for tensor in b_l1:
        tensor.deallocate()
    assert _usage(device, "model_b", ttnn.BufferType.L1) is None
    a_out.deallocate()
    assert _usage(device, "model_a", ttnn.BufferType.L1).num_buffers == 1

    for tensor in (a_l1, a_dram, unowned, l1_input):
        tensor.deallocate()


def test_memory_owner_is_per_thread(device):
    ready = threading.Event()
    release = threading.Event()
    results = {}

    def allocate_without_owner():
        ready.wait()
        results["owner"] = ttnn.current_memory_owner()
        results["tensor"] = _allocate(device, ttnn.L1_MEMORY_CONFIG)
        release.set()

    worker = threading.Thread(target=allocate_without_owner)
    worker.start()
    with ttnn.memory_owner("model_a"):
        ready.set()
        release.wait()
    worker.join()

    assert results["owner"] == ""
    assert _usage(device, "model_a", ttnn.BufferType.L1) is None
    results["tensor"].deallocate()
