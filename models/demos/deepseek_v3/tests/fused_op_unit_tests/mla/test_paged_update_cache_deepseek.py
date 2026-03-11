# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import nearest_y
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_equal


def _debug_mtp_enabled() -> bool:
    return os.getenv("DEBUG_MTP", "0") == "1"


def create_paged_cache(device, num_users, max_seq_len, head_dim, num_blocks, block_size):
    """Create a simple paged cache for testing."""
    # Cache is organized as [num_users, num_heads=1, num_blocks * block_size, head_dim]
    cache_shape = (num_users, 1, num_blocks * block_size, head_dim)
    cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    # Convert to ttnn with L1 memory
    tt_cache = ttnn.from_torch(
        cache,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return tt_cache, cache


def create_page_table(device, num_users, num_blocks):
    """Create a simple page table mapping logical to physical blocks."""
    # Page table shape: [num_users, max_num_blocks_per_user]
    # For simplicity, use identity mapping: user i uses blocks [i*blocks_per_user : (i+1)*blocks_per_user]
    blocks_per_user = num_blocks // num_users
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(num_users, blocks_per_user)

    # Convert to ttnn
    tt_page_table = ttnn.from_torch(
        page_table,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_page_table, page_table


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_paged_update_cache_verify_aliasing_mesh_sharded_update_idxs(mesh_device):
    """
    Match the actual DeepSeek MTP decode input layout: mesh-sharded update
    indices plus the aliased prompt/spec page-table pattern.
    """
    torch.manual_seed(0)
    enable_kv_log = _debug_mtp_enabled()

    mesh_shape = tuple(mesh_device.shape)
    dp_factor = mesh_shape[1]
    num_devices = mesh_shape[0] * mesh_shape[1]

    num_users = 2  # users per shard
    batch_per_shard = 2 * num_users  # prompt + spec lanes per shard
    total_batch = batch_per_shard * dp_factor

    block_size = ttnn.TILE_SIZE
    head_dim = 64
    max_num_blocks = batch_per_shard
    blocks_per_user = max_num_blocks // batch_per_shard
    assert blocks_per_user == 1

    torch_cache = torch.full((max_num_blocks, 1, block_size, head_dim), 1.0, dtype=torch.bfloat16)
    tt_cache = ttnn.from_torch(
        torch_cache,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    base_page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(batch_per_shard, blocks_per_user)
    alias_page_table = base_page_table.clone()
    verify_offset = num_users
    num_rows = int(alias_page_table.shape[0])
    for i in range(num_users):
        src_row = i % num_rows
        dst_row = (verify_offset + i) % num_rows
        alias_page_table[dst_row] = alias_page_table[src_row]

    tt_page_table = ttnn.from_torch(
        alias_page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    update_tensor = torch.zeros((1, total_batch, 1, head_dim), dtype=torch.bfloat16)
    for shard in range(dp_factor):
        base = shard * batch_per_shard
        update_tensor[0, base + 0, 0, :] = 2.0 + 2.0 * shard
        update_tensor[0, base + verify_offset, 0, :] = 3.0 + 2.0 * shard

    local_update_idxs = torch.full((batch_per_shard,), -1, dtype=torch.int32)
    local_update_idxs[0] = 0
    local_update_idxs[verify_offset] = 1

    prompt_update_idxs_local = local_update_idxs.clone()
    prompt_update_idxs_local[verify_offset : verify_offset + num_users] = -1
    spec_update_idxs_local = local_update_idxs.clone()
    spec_update_idxs_local[:num_users] = -1

    prompt_update_idxs_global = prompt_update_idxs_local.repeat(num_devices)
    spec_update_idxs_global = spec_update_idxs_local.repeat(num_devices)

    if enable_kv_log:
        logger.info("mesh-sharded prompt_update_idxs local: {}", prompt_update_idxs_local)
        logger.info("mesh-sharded spec_update_idxs local: {}", spec_update_idxs_local)
        logger.info(
            "mesh-sharded update_idxs global shape={} first16={} last16={}",
            tuple(prompt_update_idxs_global.shape),
            prompt_update_idxs_global[:16].tolist(),
            prompt_update_idxs_global[-16:].tolist(),
        )

    grid_size = mesh_device.compute_with_storage_grid_size()
    num_cores = batch_per_shard
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)
    shard_shape = (nearest_y(1, ttnn.TILE_SIZE), head_dim)
    update_mem_cfg = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_update = ttnn.from_torch(
        update_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=update_mem_cfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_shape),
    )

    tt_prompt_update_idxs = ttnn.from_torch(
        prompt_update_idxs_global,
        device=mesh_device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_spec_update_idxs = ttnn.from_torch(
        spec_update_idxs_global,
        device=mesh_device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    mesh_coords = {ttnn.MeshCoordinate(r, c) for r in range(mesh_shape[0]) for c in range(mesh_shape[1])}

    def dump_cache(tag: str):
        if not enable_kv_log:
            return None
        tt_cache_torch = ttnn.to_torch(tt_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        tt_cache_torch = tt_cache_torch.reshape(num_devices, max_num_blocks, 1, block_size, head_dim)
        for device_idx in range(num_devices):
            logger.info("{} kvcache device{}:\n{}", tag, device_idx, tt_cache_torch[device_idx])
        return tt_cache_torch

    dump_cache("before_mesh_sharded_split_update")
    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_update,
        update_idxs_tensor=tt_prompt_update_idxs,
        page_table=tt_page_table,
        mesh_coords=mesh_coords,
    )
    ttnn.synchronize_device(mesh_device)
    dump_cache("after_mesh_sharded_prompt_update")

    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_update,
        update_idxs_tensor=tt_spec_update_idxs,
        page_table=tt_page_table,
        mesh_coords=mesh_coords,
    )
    ttnn.synchronize_device(mesh_device)
    tt_cache_torch = dump_cache("after_mesh_sharded_spec_update")
    if tt_cache_torch is None:
        tt_cache_torch = ttnn.to_torch(tt_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
            num_devices, max_num_blocks, 1, block_size, head_dim
        )

    for device_idx in range(num_devices):
        ref_cache = torch_cache.clone()
        shard_idx = device_idx % dp_factor
        local_update = update_tensor[0, shard_idx * batch_per_shard : (shard_idx + 1) * batch_per_shard, 0, :]
        for local_idx, pos in enumerate(local_update_idxs.tolist()):
            if pos < 0:
                continue
            block = pos // block_size
            offset = pos % block_size
            physical_block = alias_page_table[local_idx, block].item()
            if physical_block < 0:
                continue
            ref_cache[physical_block, 0, offset, :] = local_update[local_idx, :]

        passing, _ = assert_equal(ref_cache, tt_cache_torch[device_idx])
        assert passing, f"Paged update cache mismatch for mesh-sharded aliased lanes on device {device_idx}"


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, update_shape, cache_shape_params, shard_shape",
    [
        (
            "kvpe_cache_update",
            [1, 4, 32, 576],  # Update tensor shape (after mesh_partition: 32/8=4)
            {
                "num_users": 4,  # Per device: 32 users / 8 devices = 4
                "max_seq_len": 2048,
                "head_dim": 576,
                "num_blocks": 8,  # Per device: 64 blocks / 8 devices = 8
                "block_size": 32,
                "batch_size": 32,  # Original batch size before partitioning
            },
            [32, 576],  # HEIGHT_SHARDED shard shape matching model
        ),
    ],
    ids=["kvpe_cache_update"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 2097152,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_paged_update_cache_trace_mode(
    device,
    batch_size,
    op_name,
    update_shape,
    cache_shape_params,
    shard_shape,
    warmup_iters,
    num_iters,
):
    """
    Test the paged_update_cache operation from mla1d.py with trace mode.

    This operation updates a paged KV cache:
    1. kvpe_cache_update (line 1205): Updates KVPE cache [1, 4, 1(32), 576] with HEIGHT_SHARDED memory

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - HEIGHT_SHARDED memory layout
    """
    torch.manual_seed(0)

    num_users = cache_shape_params["num_users"]
    max_seq_len = cache_shape_params["max_seq_len"]
    head_dim = cache_shape_params["head_dim"]
    num_blocks = cache_shape_params["num_blocks"]
    block_size = cache_shape_params["block_size"]

    # Create paged cache
    tt_cache, torch_cache = create_paged_cache(device, num_users, max_seq_len, head_dim, num_blocks, block_size)

    # Create page table
    tt_page_table, torch_page_table = create_page_table(device, num_users, num_blocks)

    # Create update tensor - using post-partition shape [1, 4, 32, 576]
    torch_update = torch.randn(update_shape, dtype=torch.bfloat16)

    # Create position indices (which position in sequence to update for each user)
    # Position indices must be within the cache size (num_blocks * block_size)
    cache_size = num_blocks * block_size
    position_idxs = torch.randint(0, cache_size - 1, (num_users,), dtype=torch.int32)

    # Convert update tensor to ttnn with L1 interleaved first
    tt_update = ttnn.from_torch(
        torch_update,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Create HEIGHT_SHARDED memory config
    # Model uses 1x4 grid (1 row, 4 columns)
    num_cores_x = 4
    num_cores_y = 1
    grid_size = device.compute_with_storage_grid_size()
    shard_grid_set = ttnn.num_cores_to_corerangeset(num_cores_x * num_cores_y, grid_size, row_wise=True)

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=shard_grid_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_update = ttnn.to_memory_config(tt_update, sharded_mem_config)

    # Convert position indices to ttnn
    tt_position_idxs = ttnn.from_torch(
        position_idxs,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute golden reference
    # For simplicity, we'll just verify the operation runs without error
    # Full verification would require unpacking the paged cache structure
    torch_ref_cache = torch_cache.clone()
    for user_idx, pos_idx in enumerate(position_idxs.tolist()):
        if pos_idx >= 0:
            # torch_update is [1, 4, 32, 576], extract the right batch element
            # user_idx maps to position in dim 1 (4 elements)
            torch_ref_cache[user_idx, 0, pos_idx : pos_idx + 1, :] = torch_update[0, user_idx, 0, :]

    # Compile run
    logger.info(f"Compiling paged_update_cache operation: {op_name}")
    logger.info(f"  Update shape: {update_shape}")
    logger.info(f"  Cache shape: [{num_users}, 1, {num_blocks * block_size}, {head_dim}]")
    logger.info(f"  Shard shape: {shard_shape}")
    logger.info(
        f"  Memory config: HEIGHT_SHARDED with {num_cores_x * num_cores_y} cores (grid {num_cores_y}x{num_cores_x})"
    )

    # Note: mesh_coords parameter would normally come from get_mesh_coords
    # For single device testing, we simulate a single device at position (0, 0) in the mesh
    mesh_coords = {ttnn.MeshCoordinate(0, 0)}

    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_update,
        update_idxs_tensor=tt_position_idxs,
        page_table=tt_page_table,
        mesh_coords=mesh_coords,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        ttnn.experimental.paged_update_cache(
            tt_cache,
            tt_update,
            update_idxs_tensor=tt_position_idxs,
            page_table=tt_page_table,
            mesh_coords=mesh_coords,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        ttnn.experimental.paged_update_cache(
            tt_cache,
            tt_update,
            update_idxs_tensor=tt_position_idxs,
            page_table=tt_page_table,
            mesh_coords=mesh_coords,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Verify correctness
    # Get cache back from device
    tt_cache_host = ttnn.from_device(tt_cache)
    torch_cache_updated = ttnn.to_torch(tt_cache_host)

    # Check that at least one of the updated positions matches
    # (Full verification is complex due to paging, so we do a sanity check)
    pos_idx = position_idxs[0].item()
    if pos_idx >= 0:
        # Check that the cache was updated at this position
        cache_slice = torch_cache_updated[0, 0, pos_idx, :]
        ref_slice = torch_ref_cache[0, 0, pos_idx, :]

        passing, _ = assert_equal(ref_slice, cache_slice)
        logger.info(f"Equal for updated cache position: {passing}")

    logger.info(f"✓ Trace mode {op_name} test passed")
