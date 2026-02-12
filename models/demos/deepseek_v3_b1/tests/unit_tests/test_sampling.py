# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.sampling.op import SamplingOp


def _run_sampling_argmax_single_device_101_cores(device, seed: int, final_core_idx: int):
    grid_size = device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    if len(all_device_cores) < 101:
        pytest.skip(f"Need at least 101 cores, found {len(all_device_cores)}")

    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]

    num_cores = len(active_cores)
    scores_shape = (1, 160 * num_cores)
    input_shard_shape = (1, 160)
    output_shape = (1, 1)
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        f"Testing sampling argmax: single-device/101-cores, seed={seed}, final_core_idx={final_core_idx}, "
        "160 values per core"
    )

    torch.manual_seed(seed)
    torch_scores = torch.randn(scores_shape, dtype=torch.bfloat16)
    torch_indices = torch.arange(scores_shape[1], dtype=torch.int32).reshape(scores_shape)

    torch_expected_idx = SamplingOp.golden(torch_scores, torch_indices, k=1, p=1.0)

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(
        final_core_grid,
        output_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    ttnn_scores = ttnn.from_torch(
        torch_scores,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )

    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    ttnn_output_index = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=1,
        p=1.0,
        final_core_coord=final_core,
        final_mesh_coord=None,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == output_shape, f"Expected output shape {output_shape}, got {output_torch.shape}"
    logger.info(f"Golden output: {output_torch}")
    logger.info(f"Golden expected index: {torch_expected_idx}")
    assert torch.equal(
        output_torch.to(torch.uint32), torch_expected_idx
    ), f"Argmax index mismatch. expected={torch_expected_idx.item()}, got={output_torch.item()}"

    logger.info(
        f"Sampling argmax test passed. seed={seed}, final_core_idx={final_core_idx}, index={int(output_torch.item())}"
    )


@pytest.mark.parametrize(
    "seed, final_core_idx",
    [
        (2005, 100),  # last active core (original behavior)
        (17, 0),  # first active core
        (1337, 50),  # middle active core
        (4242, 73),  # non-boundary core
    ],
)
def test_sampling_argmax_single_device_101_cores(device, seed, final_core_idx):
    """
    Test k=1 sampling (argmax path) for a single device and 101 cores.
    Covers multiple random seeds and different final-core placements.
    """
    _run_sampling_argmax_single_device_101_cores(device, seed=seed, final_core_idx=final_core_idx)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "final_mesh_coord,seed,final_core_idx",
    [
        ((0, 0), 2005, 100),
        ((0, 1), 520, 0),
        ((1, 0), 1337, 50),
        ((1, 1), 4242, 73),
    ],
)
def test_sampling_argmax_mesh_2x2_axis_x(mesh_device, final_mesh_coord, seed, final_core_idx):
    """
    Mesh extension test:
    - per-device local 101-core argmax reduction,
    - stage-1 axis-x reduction across row pairs,
    - stage-2 reduction to parameterized final mesh coord.
    """
    grid_size = mesh_device.compute_with_storage_grid_size()
    all_device_cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    if len(all_device_cores) < 101:
        pytest.skip(f"Need at least 101 cores, found {len(all_device_cores)}")

    active_cores = all_device_cores[:101]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in active_cores})
    assert 0 <= final_core_idx < len(active_cores), f"final_core_idx={final_core_idx} out of range"
    final_core = active_cores[final_core_idx]

    num_devices = 4
    num_cores = len(active_cores)
    scores_shape_per_device = (1, 160 * num_cores)
    input_shard_shape = (1, 160)
    output_shape_per_device = (1, 1)
    scratch_shape_per_device = (1, 16)  # 4 slots x 16B / 4B(uint32) = 16 uint32 entries
    tile_1x32 = ttnn.Tile([1, 32])

    logger.info(
        "Testing sampling argmax mesh(2x2): "
        f"seed={seed}, final_core_idx={final_core_idx}, final_mesh_coord={final_mesh_coord}"
    )
    torch.manual_seed(seed)

    # Build all device inputs in one shot, then shard device dimension with mesh mapper.
    torch_scores_all = torch.randn((num_devices, *scores_shape_per_device), dtype=torch.bfloat16)
    # Make global index space unique across devices for deterministic tie-break checks.
    torch_indices_all = torch.arange(num_devices * scores_shape_per_device[1], dtype=torch.int32).reshape(
        num_devices, *scores_shape_per_device
    )
    torch_expected_idx = SamplingOp.golden(
        torch_scores_all.reshape(1, -1), torch_indices_all.reshape(1, -1), k=1, p=1.0
    )

    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    final_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(final_core, final_core)})
    output_shard_spec = ttnn.ShardSpec(
        final_core_grid,
        output_shape_per_device,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )
    scratch_shard_spec = ttnn.ShardSpec(
        final_core_grid,
        scratch_shape_per_device,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        scratch_shard_spec,
    )

    mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    ttnn_scores = ttnn.from_torch(
        torch_scores_all,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=input_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, *output_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(mesh_device, final_core_grid, 0)
    ttnn.synchronize_device(mesh_device)

    ttnn_result = SamplingOp.op(
        scores_tensor=ttnn_scores,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        k=1,
        p=1.0,
        final_core_coord=final_core,
        final_mesh_coord=final_mesh_coord,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        mesh_axis="x",
    )
    ttnn.synchronize_device(mesh_device)

    output_shards = ttnn.get_device_tensors(ttnn_result)
    final_device_idx = final_mesh_coord[0] * 2 + final_mesh_coord[1]
    final_output_torch = ttnn.to_torch(output_shards[final_device_idx])
    final_output_index = final_output_torch.to(torch.uint32).reshape(1, 1)

    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={int(final_output_index.item())}"
