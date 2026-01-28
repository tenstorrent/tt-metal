# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import pytest


def test_as_tensor(device):
    # Create a Torch tensor and convert it to a TT-NN tensor
    tensor = ttnn.as_tensor(torch.randn((2, 3)), dtype=ttnn.bfloat16)
    logger.info(tensor.shape)


def test_from_torch(device):
    # Create a Torch tensor and convert it to a TT-NN tensor
    torch_tensor = torch.randn((4, 5), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=device)

    logger.info("TT-NN tensor shape", ttnn_tensor.shape)


def test_to_torch(device):
    # Create a TT-NN tensor and convert it to a Torch tensor
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    torch_tensor = ttnn.to_torch(ttnn_tensor)

    logger.info("Torch tensor shape", torch_tensor.shape)


def test_to_device(device):
    # Open the device
    # device_id = 0
    # device = ttnn.open_device(device_id=device_id)

    # Create a TT-NN tensor and move it to the specified device
    tensor_on_host = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=ttnn.bfloat16)
    ttnn_tensor = ttnn.to_device(tensor_on_host, device=device)

    logger.info("TT-NN tensor shape after moving to device", ttnn_tensor.shape)


def test_from_device(device):
    # Open the device
    # device_id = 0
    # device = ttnn.open_device(device_id=device_id)

    # Create a TT-NN tensor on the device and move it back to the host
    ttnn_tensor_on_device = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_on_host = ttnn.from_device(ttnn_tensor_on_device)

    logger.info("TT-NN tensor shape after moving from device to host", ttnn_tensor_on_host.shape)


def test_to_layout(device):
    # Create a TT-NN tensor and change its layout
    ttnn_tensor = ttnn.rand((2, 3, 4), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_layout_changed = ttnn.to_layout(ttnn_tensor, layout=ttnn.TILE_LAYOUT)

    logger.info("TT-NN tensor shape after changing layout", ttnn_tensor_layout_changed.shape)


def test_dump_tensor(device):
    # Create a TT-NN tensor and dump its contents
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.dump_tensor(file_name="ttnn_tensor.tensorbin", tensor=ttnn_tensor)


def test_load_tensor(device):
    # Create a TT-NN tensor, dump its contents, and load it back
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.dump_tensor(file_name="ttnn_tensor.tensorbin", tensor=ttnn_tensor)

    loaded_tensor = ttnn.load_tensor(file_name="ttnn_tensor.tensorbin", device=device)
    logger.info("Loaded TT-NN tensor shape", loaded_tensor.shape)


def test_deallocate(device):
    # Create a TT-NN tensor and deallocate its memory
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn.deallocate(ttnn_tensor)


def test_reallocate(device):
    # Create a TT-NN tensor, deallocate it, and then reallocate it
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_reallocated = ttnn.reallocate(ttnn_tensor)

    logger.info("Reallocated TT-NN tensor shape", ttnn_tensor_reallocated.shape)


def test_to_memory_config(device):
    # Create a TT-NN tensor and change its memory configuration
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    ttnn_tensor_memory_config_changed = ttnn.to_memory_config(ttnn_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    logger.info("TT-NN tensor shape after changing memory configuration", ttnn_tensor_memory_config_changed.shape)


def test_copy_device_to_host_tensor(device):
    # Create a TT-NN tensor and copy it to the host
    ttnn_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)
    ttnn.copy_device_to_host_tensor(ttnn_tensor, host_tensor)

    logger.info("Host tensor shape after copying from device", host_tensor.shape)


def test_copy_host_to_device_tensor(device):
    # Create a host tensor and copy it to a pre-allocated device tensor
    dtype = ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT

    host_tensor = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=dtype, layout=layout)

    device_tensor = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
    ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    # Verify round-trip
    roundtrip_host = ttnn.from_device(device_tensor)
    assert torch.allclose(ttnn.to_torch(roundtrip_host), ttnn.to_torch(host_tensor))
    logger.info("TT-NN tensor shape after copying to device", device_tensor.shape)


def test_to_dtype(device):
    tensor = ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16))
    tensor = ttnn.to_dtype(tensor, dtype=ttnn.uint8)
    assert tensor.dtype == ttnn.uint8
    assert tensor.shape == (10, 64, 32)

    logger.info("TT-NN tensor shape after converting to uint8", tensor.shape)


def test_to_dtype_nd_sharded(device):
    # Create an ND-sharded tensor of shape [10, 64, 32] (device), then bring it back to host (same dtype)
    host_src = ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    shard_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    nd_shard_spec = ttnn.NdShardSpec((10, 64, 32), shard_cores, ttnn.ShardOrientation.ROW_MAJOR)
    device_tensor = ttnn.to_device(
        host_src, device=device, memory_config=ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)
    )
    tensor = ttnn.from_device(device_tensor)
    tensor = ttnn.to_dtype(tensor, dtype=ttnn.uint8)
    assert tensor.dtype == ttnn.uint8
    assert tensor.shape == (10, 64, 32)

    logger.info("TT-NN tensor shape after converting to uint16", tensor.shape)


@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.int32, ttnn.uint32, ttnn.uint16]
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_copy_host_to_device_tensor_pytest(dtype, layout, device):
    # Create a host tensor and copy it to a pre-allocated device tensor
    # shape = (2, 3)
    # torch_tensor = torch.randn(shape, dtype=torch.int32)
    host_tensor = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=dtype, layout=layout)
    # host_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)

    device_tensor = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
    ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    # Verify round-trip
    roundtrip_host = ttnn.from_device(device_tensor)
    assert torch.allclose(ttnn.to_torch(roundtrip_host), ttnn.to_torch(host_tensor))


def test_copy_host_to_device_tensor_sharded_pytest(device):
    # Create a sharded TT-NN tensor (height-sharded) and copy it to the host
    shape = (1, 2, 128, 128)
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    # Height-sharded across 2 cores with shard shape [64, 128]
    # shard_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
    # shard_spec = ttnn.ShardSpec(shard_cores, (128, 128), ttnn.ShardOrientation.ROW_MAJOR)
    # memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    # Height-sharded across cores with ND sharding (shard shape [N=1, H=64, W=128])
    shard_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec((2, 64, 128), shard_cores, ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)

    # Use from_torch to avoid on-device typecast with sharded layout
    host_tensor = ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=layout, memory_config=memory_config)
    device_tensor = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
    ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    roundtrip_device = ttnn.from_device(device_tensor)
    assert torch.allclose(ttnn.to_torch(roundtrip_device), ttnn.to_torch(device_tensor))

    logger.info("Host tensor (sharded) shape after copying from device", host_tensor.shape)


@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16]
)  # ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])  # , ttnn.TILE_LAYOUT])
def test_copy_device_to_host_tensor_pytest(dtype, layout, device):
    # Create a TT-NN tensor and copy it to the host
    # ttnn_tensor = ttnn.rand((2, 3), dtype=dtype, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_tensor = torch.randn((2, 3), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)
    ttnn.copy_device_to_host_tensor(ttnn_tensor, host_tensor)
    assert torch.allclose(ttnn.to_torch(host_tensor, dtype=torch.bfloat16), torch_tensor)
    logger.info("Host tensor shape after copying from device", host_tensor.shape)


def test_copy_device_to_host_tensor_sharded_pytest(device):
    # Create a sharded TT-NN tensor (height-sharded) and copy it to the host
    shape = (1, 2, 128, 128)
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    # Height-sharded across 2 cores with shard shape [64, 128]
    shard_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
    shard_spec = ttnn.ShardSpec(shard_cores, (256, 128), ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    # Height-sharded across cores with ND sharding (shard shape [N=1, H=64, W=128])
    # shard_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    # nd_shard_spec = ttnn.NdShardSpec((2, 64, 128), shard_cores, ttnn.ShardOrientation.ROW_MAJOR)
    # memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)

    # Use from_torch to avoid on-device typecast with sharded layout
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)
    ttnn.copy_device_to_host_tensor(ttnn_tensor, host_tensor)
    assert torch.equal(ttnn.to_torch(host_tensor), torch_tensor)
    logger.info("Host tensor (sharded) shape after copying from device", host_tensor.shape)
