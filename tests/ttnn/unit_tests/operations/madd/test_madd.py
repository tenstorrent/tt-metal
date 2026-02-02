# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

pytestmark = pytest.mark.use_module_device


# @pytest.mark.parametrize("dim1", [32])
# @pytest.mark.parametrize("dim2", [32])
@pytest.mark.parametrize("dim1", [1024, 128, 64, 32])
@pytest.mark.parametrize("dim2", [1024, 128, 64, 32])
def test_madd_interleaved(device, dim1, dim2):
    # device.disable_and_clear_program_cache()

    torch.manual_seed(0)

    shape = (dim1, dim2)

    memory_config = ttnn.DRAM_MEMORY_CONFIG
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_c = torch.rand(shape, dtype=torch.bfloat16)
    # torch_a = torch.ones(shape, dtype=torch.bfloat16)
    # torch_b = torch.zeros(shape, dtype=torch.bfloat16)
    # torch_c = torch.zeros(shape, dtype=torch.bfloat16) + 2
    torch_output_tensor = torch_a * torch_b + torch_c

    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    # print(ttnn_a)
    # print(ttnn_b)
    # print(ttnn_c)

    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config, compute_kernel_config=compute_config)
    # print(output_tensor)

    assert output_tensor.shape == shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


# Height sharded: shards are distributed along the height dimension
@pytest.mark.parametrize(
    "shard_shape, core_grid_shape",
    [
        ((64, 128), (2, 4)),  # 2x4=8 cores, 8 * 64 height = 512
        ((32, 64), (1, 4)),  # 1x4=4 cores, 4 * 32 height = 128
    ],
    ids=["64x128_2x4cores", "32x64_1x4cores"],
)
def test_madd_sharded_height(device, shard_shape, core_grid_shape):
    """Test madd with height-sharded tensors. All tensors must have identical shard configuration."""
    torch.manual_seed(0)

    # Derive tensor shape from shard shape and core grid
    # For height sharding: tensor_height = shard_height * num_cores, tensor_width = shard_width
    shard_height, shard_width = shard_shape
    grid_x, grid_y = core_grid_shape
    num_cores = grid_x * grid_y
    shape = (shard_height * num_cores, shard_width)

    # Create core grid from grid shape
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (grid_x - 1, grid_y - 1))})

    memory_config = ttnn.create_sharded_memory_config(
        shape=list(shard_shape),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_c = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_a * torch_b + torch_c

    # All three tensors use the same sharded memory config
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config)

    assert output_tensor.shape == shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


# Width sharded: shards are distributed along the width dimension
@pytest.mark.parametrize(
    "shard_shape, core_grid_shape",
    [
        ((512, 32), (1, 4)),  # 1x4=4 cores, 4 * 32 width = 128
        ((256, 64), (2, 2)),  # 2x2=4 cores, 4 * 64 width = 256
    ],
    ids=[
        "512x32_1x4cores",
        "256x64_2x2cores",
    ],
)
def test_madd_sharded_width(device, shard_shape, core_grid_shape):
    """Test madd with width-sharded tensors. All tensors must have identical shard configuration."""
    torch.manual_seed(0)

    # Derive tensor shape from shard shape and core grid
    # For width sharding: tensor_height = shard_height, tensor_width = shard_width * num_cores
    shard_height, shard_width = shard_shape
    grid_x, grid_y = core_grid_shape
    num_cores = grid_x * grid_y
    shape = (shard_height, shard_width * num_cores)

    # Create core grid from grid shape
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (grid_x - 1, grid_y - 1))})

    memory_config = ttnn.create_sharded_memory_config(
        shape=list(shard_shape),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_c = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_a * torch_b + torch_c

    # All three tensors use the same sharded memory config
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config)

    assert output_tensor.shape == shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


# Block sharded: shards are distributed in a 2D grid pattern
@pytest.mark.parametrize(
    "shard_shape, core_grid_shape",
    [
        ((128, 64), (4, 2)),  # 4x2=8 cores, 4*128 height = 512, 2*64 width = 128
        ((64, 64), (2, 2)),  # 2x2=4 cores, 2*64 height = 128, 2*64 width = 128
    ],
    ids=["128x64_4x2cores", "64x64_2x2cores"],
)
def test_madd_sharded_block(device, shard_shape, core_grid_shape):
    """Test madd with block-sharded tensors. All tensors must have identical shard configuration."""
    torch.manual_seed(0)

    # Derive tensor shape from shard shape and core grid
    # For block sharding: tensor_height = shard_height * grid_x, tensor_width = shard_width * grid_y
    shard_height, shard_width = shard_shape
    grid_x, grid_y = core_grid_shape
    shape = (shard_height * grid_y, shard_width * grid_x)

    print(f"{grid_x=}, {grid_y=}, {shape=}, {shard_shape=}")
    # Create core grid from grid shape
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (grid_x - 1, grid_y - 1))})

    memory_config = ttnn.create_sharded_memory_config(
        shape=list(shard_shape),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_c = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_a * torch_b + torch_c

    # All three tensors use the same sharded memory config
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config)

    assert output_tensor.shape == shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


# @pytest.mark.parametrize("h", [16*8*32*11*10])
# @pytest.mark.parametrize("w", [32])
# @pytest.mark.parametrize("count", [25])
# def test_perf_interleaved(device, h, w, count):
#     import time

#     torch.manual_seed(0)

#     shape = (h, w)

#     torch_a = torch.rand(shape, dtype=torch.bfloat16)
#     torch_b = torch.rand(shape, dtype=torch.bfloat16)
#     torch_c = torch.rand(shape, dtype=torch.bfloat16)


#     memory_config = ttnn.DRAM_MEMORY_CONFIG
#     compute_config = ttnn.init_device_compute_kernel_config(
#         device.arch(),
#         math_fidelity=ttnn.MathFidelity.LoFi,
#         math_approx_mode=True,
#         fp32_dest_acc_en=False,
#         packer_l1_acc=False,
#         dst_full_sync_en=False
#     )

#     ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
#     ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
#     ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

#     # Warm-up run
#     start_time = time.time()
#     output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config, compute_kernel_config=compute_config)
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Warmup iteration for shape {shape}: {total_time * 1000:.3f} ms")

#     # Run
#     start_time = time.time()
#     for _ in range(count):
#         output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config, compute_kernel_config=compute_config)
#     end_time = time.time()
#     total_time = end_time - start_time
#     avg_time_per_iter = total_time / count
#     print(f"Average time per madd iteration for shape {shape}: {avg_time_per_iter * 1000:.3f} ms")


# @pytest.mark.parametrize("h", [16*8*32*11*10])
# @pytest.mark.parametrize("w", [32])
# @pytest.mark.parametrize("core_grid_shape", [(11, 10)])
# @pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT])
# @pytest.mark.parametrize("count", [25])
# def test_perf_sharded(device, h, w, core_grid_shape, strategy, count):
#     import time

#     torch.manual_seed(0)

#     shape = (h, w)
#     grid_x, grid_y = core_grid_shape
#     num_cores = grid_x * grid_y

#     # Calculate shard shape based on strategy
#     if strategy == ttnn.ShardStrategy.HEIGHT:
#         shard_shape = (h // num_cores, w)
#     elif strategy == ttnn.ShardStrategy.WIDTH:
#         shard_shape = (h, w // num_cores)
#     elif strategy == ttnn.ShardStrategy.BLOCK:
#         shard_shape = (h // grid_y, w // grid_x)

#     # Create core grid
#     core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (grid_x - 1, grid_y - 1))})

#     memory_config = ttnn.create_sharded_memory_config(
#         shape=list(shard_shape),
#         core_grid=core_grid,
#         strategy=strategy,
#         orientation=ttnn.ShardOrientation.ROW_MAJOR,
#         use_height_and_width_as_shard_shape=True,
#     )
#     compute_config = ttnn.init_device_compute_kernel_config(
#         device.arch(),
#         math_fidelity=ttnn.MathFidelity.LoFi,  # Match interleaved test for fair comparison
#         math_approx_mode=True,
#         fp32_dest_acc_en=False,
#         packer_l1_acc=False,
#         dst_full_sync_en=False
#     )

#     torch_a = torch.rand(shape, dtype=torch.bfloat16)
#     torch_b = torch.rand(shape, dtype=torch.bfloat16)
#     torch_c = torch.rand(shape, dtype=torch.bfloat16)

#     ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
#     ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
#     ttnn_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

#     # Warm-up run
#     start_time = time.time()
#     output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config, compute_kernel_config=compute_config)
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Warmup iteration for shape {shape} ({strategy.name} sharded): {total_time * 1000:.3f} ms")

#     # Run
#     start_time = time.time()
#     for _ in range(count):
#         output_tensor = ttnn.madd(ttnn_a, ttnn_b, ttnn_c, memory_config=memory_config, compute_kernel_config=compute_config)
#     end_time = time.time()
#     total_time = end_time - start_time
#     avg_time_per_iter = total_time / count
#     print(f"Average time per madd iteration for shape {shape} ({strategy.name} sharded): {avg_time_per_iter * 1000:.3f} ms")

#     print(device.compute_with_storage_grid_size())
#     print(f"Architecture: {ttnn.get_arch_name()}")
