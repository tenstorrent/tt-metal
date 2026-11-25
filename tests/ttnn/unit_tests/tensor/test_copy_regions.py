# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import ttnn

# Bytes of one 32x32 tile page in a DRAM interleaved TILE_LAYOUT buffer.
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
NUM_BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM = 6, 2, 64, 128


def _paged_cache(device, dtype, seed):
    torch.manual_seed(seed)
    data = torch.randn(NUM_BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    return ttnn.from_torch(
        data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _block_bytes(dtype):
    # A paged cache is tiled block by block, so one block is one byte range of the buffer.
    return HEADS * (BLOCK_SIZE // 32) * (HEAD_DIM // 32) * TILE_BYTES[dtype]


def _all_bytes(tensor, dtype):
    out = np.zeros(NUM_BLOCKS * _block_bytes(dtype), dtype=np.uint8)
    ttnn.copy_device_to_host_regions(tensor, [(0, 0, out.nbytes, out.ctypes.data)])
    return out


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("form", ["tuples", "array"])
def test_blocks_round_trip_through_host(device, dtype, form):
    src = _paged_cache(device, dtype, seed=0)
    dst = _paged_cache(device, dtype, seed=1)
    bb = _block_bytes(dtype)
    src_bytes, dst_before = _all_bytes(src, dtype), _all_bytes(dst, dtype)

    # Blocks 4 and 1 of src go to host slots 0 and 1, then to blocks 0 and 5 of dst.
    host = np.zeros(2 * bb, dtype=np.uint8)
    base = host.ctypes.data
    reads = [(0, 4 * bb, bb, base), (0, 1 * bb, bb, base + bb)]
    writes = [(0, 0 * bb, bb, base), (0, 5 * bb, bb, base + bb)]
    if form == "array":
        reads, writes = np.array(reads, dtype=np.uint64), np.array(writes, dtype=np.uint64)
    ttnn.copy_device_to_host_regions(src, reads)
    ttnn.copy_host_to_device_regions(dst, writes)

    dst_after = _all_bytes(dst, dtype)
    expected = dst_before.copy()
    expected[0:bb] = src_bytes[4 * bb : 5 * bb]
    expected[5 * bb : 6 * bb] = src_bytes[1 * bb : 2 * bb]
    np.testing.assert_array_equal(dst_after, expected)


def test_non_blocking_read_lands_after_synchronize(device):
    dtype = ttnn.bfloat8_b
    src = _paged_cache(device, dtype, seed=2)
    bb = _block_bytes(dtype)
    expected = _all_bytes(src, dtype)[2 * bb : 3 * bb]

    host = np.zeros(bb, dtype=np.uint8)
    regions = np.array([(0, 2 * bb, bb, host.ctypes.data)], dtype=np.uint64)
    ttnn.copy_device_to_host_regions(src, regions, blocking=False)
    ttnn.synchronize_device(device)
    np.testing.assert_array_equal(host, expected)


def test_page_by_page_regions_equal_one_region(device):
    dtype = ttnn.bfloat8_b
    src = _paged_cache(device, dtype, seed=3)
    page, bb = TILE_BYTES[dtype], _block_bytes(dtype)
    whole = _all_bytes(src, dtype)

    # One region per tile page of blocks 1..3, back to back in host memory: they merge into one transfer.
    host = np.zeros(3 * bb, dtype=np.uint8)
    pages = np.arange(3 * bb // page, dtype=np.uint64)
    regions = np.stack(
        [np.zeros_like(pages), bb + pages * page, np.full_like(pages, page), host.ctypes.data + pages * page], axis=1
    )
    ttnn.copy_device_to_host_regions(src, np.ascontiguousarray(regions))
    np.testing.assert_array_equal(host, whole[bb : 4 * bb])


def test_rejected_batch_enqueues_nothing(device, expect_error):
    dtype = ttnn.bfloat8_b
    src = _paged_cache(device, dtype, seed=5)
    page, bb = TILE_BYTES[dtype], _block_bytes(dtype)
    whole = _all_bytes(src, dtype)

    # A valid first region followed by a bad one: the call fails and the first region is not read either.
    stale = np.zeros(bb, dtype=np.uint8)
    with expect_error(RuntimeError, "whole number of"):
        ttnn.copy_device_to_host_regions(src, [(0, 0, bb, stale.ctypes.data), (0, page // 2, page, stale.ctypes.data)])
    with expect_error(RuntimeError, "outside the"):
        ttnn.copy_device_to_host_regions(
            src, [(0, 0, bb, stale.ctypes.data), (0, NUM_BLOCKS * bb, bb, stale.ctypes.data)]
        )
    with expect_error(RuntimeError, "out of range"):
        ttnn.copy_device_to_host_regions(src, np.array([(2**32, 0, bb, stale.ctypes.data)], dtype=np.uint64))

    # The queue still works, and nothing from the rejected calls lands in their host memory.
    host = np.zeros(bb, dtype=np.uint8)
    ttnn.copy_device_to_host_regions(src, [(0, 3 * bb, bb, host.ctypes.data)])
    ttnn.synchronize_device(device)
    np.testing.assert_array_equal(host, whole[3 * bb : 4 * bb])
    assert not stale.any()


def _device_bytes(tensor, mesh_device):
    # Each device's buffer bytes through the ordinary tensor.cpu() read, in row-major mesh order (the shard index).
    shards = tensor.cpu().host_buffer()
    rows, cols = tuple(mesh_device.shape)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]
    return [np.frombuffer(np.from_dlpack(shards.get_shard(c)).tobytes(), dtype=np.uint8) for c in coords]


@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 4)], ids=["1x2", "2x4"], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_regions_address_one_device_of_a_mesh(mesh_device, dtype, expect_error):
    n, bb = mesh_device.get_num_devices(), _block_bytes(dtype)
    torch.manual_seed(4)
    data = torch.randn(n * NUM_BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    cache = ttnn.from_torch(
        data,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0),
    )
    before = _device_bytes(cache, mesh_device)
    assert len({b.tobytes() for b in before}) == n, "precondition: every device holds different bytes"

    # Block 2 of device 1 and block 4 of the last device to host.
    host = np.zeros(2 * bb, dtype=np.uint8)
    ttnn.copy_device_to_host_regions(
        cache, [(1, 2 * bb, bb, host.ctypes.data), (n - 1, 4 * bb, bb, host.ctypes.data + bb)]
    )
    np.testing.assert_array_equal(host[:bb], before[1][2 * bb : 3 * bb])
    np.testing.assert_array_equal(host[bb:], before[n - 1][4 * bb : 5 * bb])

    # Device 1's block 2 into block 0 of the last device; no other device changes.
    ttnn.copy_host_to_device_regions(cache, [(n - 1, 0, bb, host.ctypes.data)])
    after = _device_bytes(cache, mesh_device)
    expected = [b.copy() for b in before]
    expected[n - 1][:bb] = before[1][2 * bb : 3 * bb]
    for device in range(n):
        np.testing.assert_array_equal(after[device], expected[device], err_msg=f"device {device}")

    with expect_error(RuntimeError, "out of range"):
        ttnn.copy_device_to_host_regions(cache, [(n, 0, bb, host.ctypes.data)])
