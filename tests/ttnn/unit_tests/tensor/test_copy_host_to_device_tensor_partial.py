# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _height_sharded_l1_config(shape, num_cores_y):
    # Use L1 so BufferPageMapping::all_cores equals the user's Tensix grid, and a CoreRangeSet
    # filter selects exactly the intended logical cores. DRAM-sharded buffers enumerate banks
    # on row y=0 only, which makes a column-oriented filter like (0,1) invalid for DRAM.
    h, w = shape[2], shape[3]
    shard_h = h // num_cores_y
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores_y - 1))})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _height_sharded_dram_config(shape, num_cores_x):
    # DRAM banks are enumerated only on row y=0, so a DRAM shard grid must live on that row.
    # Filters for DRAM-sharded buffers must target coordinates of the form (x, 0).
    h, w = shape[2], shape[3]
    shard_h = h // num_cores_x
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_copy_host_to_device_tensor_partial_complementary_filters_reconstruct_reference(device, layout):
    shape = (1, 1, 64, 32)
    mem_config = _height_sharded_l1_config(shape, num_cores_y=2)
    torch_ref = torch.randint(0, 2**16, shape, dtype=torch.uint32)
    torch_sent = torch.full(shape, 0xABCD1234, dtype=torch.uint32)

    tt_ref = ttnn.from_torch(torch_ref, dtype=ttnn.uint32, layout=layout, device=None)
    tt_sent = ttnn.from_torch(torch_sent, dtype=ttnn.uint32, layout=layout, device=None)

    dev_staged = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, layout, device, mem_config)
    dev_ref = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, layout, device, mem_config)

    ttnn.copy_host_to_device_tensor(tt_sent, dev_staged)
    f0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    f1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1))})
    ttnn.copy_host_to_device_tensor_partial(tt_ref, dev_staged, f0)
    ttnn.copy_host_to_device_tensor_partial(tt_ref, dev_staged, f1)

    ttnn.copy_host_to_device_tensor(tt_ref, dev_ref)

    got = ttnn.to_torch(ttnn.from_device(dev_staged))
    exp = ttnn.to_torch(ttnn.from_device(dev_ref))
    assert torch.equal(got, exp)


def test_copy_host_to_device_tensor_partial_empty_filter_is_noop(device):
    shape = (1, 1, 64, 32)
    mem_config = _height_sharded_l1_config(shape, num_cores_y=2)
    torch_sent = torch.full(shape, 0x55AA55AA, dtype=torch.uint32)
    torch_new = torch.full(shape, 0x11223344, dtype=torch.uint32)

    tt_sent = ttnn.from_torch(torch_sent, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)
    tt_new = ttnn.from_torch(torch_new, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)

    dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    ttnn.copy_host_to_device_tensor(tt_sent, dev)
    ttnn.copy_host_to_device_tensor_partial(tt_new, dev, ttnn.CoreRangeSet([]))
    got = ttnn.to_torch(ttnn.from_device(dev))
    assert torch.equal(got, torch_sent)


def test_copy_host_to_device_tensor_partial_full_grid_matches_full_copy(device):
    shape = (1, 1, 64, 32)
    mem_config = _height_sharded_l1_config(shape, num_cores_y=2)
    torch_ref = torch.randint(0, 2**20, shape, dtype=torch.uint32)
    tt_ref = ttnn.from_torch(torch_ref, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)

    full_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    dev_partial = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config
    )
    dev_full = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)

    torch_sent = torch.full(shape, 1, dtype=torch.uint32)
    tt_sent = ttnn.from_torch(torch_sent, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)
    ttnn.copy_host_to_device_tensor(tt_sent, dev_partial)
    ttnn.copy_host_to_device_tensor_partial(tt_ref, dev_partial, full_grid)
    ttnn.copy_host_to_device_tensor(tt_ref, dev_full)

    got = ttnn.to_torch(ttnn.from_device(dev_partial))
    exp = ttnn.to_torch(ttnn.from_device(dev_full))
    assert torch.equal(got, exp)


def test_copy_host_to_device_tensor_partial_complementary_filters_dram_sharded(device):
    # Same contract as the L1 version, but over a DRAM-sharded buffer whose shard grid lives on
    # row y=0.  Verifies that the CoreRangeSet filter correctly partitions writes for DRAM too.
    shape = (1, 1, 64, 32)
    mem_config = _height_sharded_dram_config(shape, num_cores_x=2)
    torch_ref = torch.randint(0, 2**16, shape, dtype=torch.uint32)
    torch_sent = torch.full(shape, 0xCAFEBABE, dtype=torch.uint32)

    tt_ref = ttnn.from_torch(torch_ref, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)
    tt_sent = ttnn.from_torch(torch_sent, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)

    dev_staged = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config
    )
    dev_ref = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)

    ttnn.copy_host_to_device_tensor(tt_sent, dev_staged)
    f0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    f1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
    ttnn.copy_host_to_device_tensor_partial(tt_ref, dev_staged, f0)
    ttnn.copy_host_to_device_tensor_partial(tt_ref, dev_staged, f1)

    ttnn.copy_host_to_device_tensor(tt_ref, dev_ref)

    got = ttnn.to_torch(ttnn.from_device(dev_staged))
    exp = ttnn.to_torch(ttnn.from_device(dev_ref))
    assert torch.equal(got, exp)


def test_copy_host_to_device_tensor_partial_interleaved_raises(device):
    shape = (1, 1, 32, 32)
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    tt_a = ttnn.from_torch(
        torch.ones(shape, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
    )
    tt_b = ttnn.from_torch(
        torch.full(shape, 2, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
    )
    dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem_config)
    ttnn.copy_host_to_device_tensor(tt_a, dev)
    flt = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    with pytest.raises(RuntimeError):
        ttnn.copy_host_to_device_tensor_partial(tt_b, dev, flt)
