# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Isolated unit test for OLMo fused QK-norm using ttnn.fused_rms_minimal on a (8,4) mesh.

Validates PCC against PyTorch reference for both K-norm [1,1,32,128] and
Q-norm [1,1,32,640] decode shapes, matching exactly the memory configs and
weight layouts used in llama_attention.py.

Also contains an axis-1 prototype path that shards Q/K dimensions over the 4
mesh columns for per-op validation only; it is not promoted into full model flow.

Run:
    export ARCH_NAME=wormhole_b0 && export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    pytest models/demos/olmo_galaxy/tests/test_olmo_fused_qk_norm.py -s -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

CLUSTER_SHAPE = (8, 4)  # (rows, cols) = (axis0, axis1)
N_ROW_DEVICES = CLUSTER_SHAPE[0]  # 8 devices along cluster_axis=0
N_COL_DEVICES = CLUSTER_SHAPE[1]  # 4 devices along cluster_axis=1
BATCH_PER_DEVICE = 8  # 32 users / 4 col devices
MAX_BATCH = 32
SHARD_HEIGHT = 32  # tile width used for weight reshaping

# OLMo K/Q norm dims
K_DIM_TOTAL = 1024  # 8 KV heads × 128, split over 8 row devices → 128 per device
Q_DIM_TOTAL = 5120  # 40 Q heads × 128, split over 8 row devices → 640 per device
K_DIM_PER_DEVICE = K_DIM_TOTAL // N_ROW_DEVICES  # 128
Q_DIM_PER_DEVICE = Q_DIM_TOTAL // N_ROW_DEVICES  # 640
DEVICE_PARAMS = [
    {
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        "fabric_config": True,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
]


def get_torch_rms(x, gamma, eps):
    """PyTorch reference RMS norm over last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_k_norm_configs(sub_device_crs):
    """Memory/program configs for K-norm: [1,1,32,128] on 2 cores (same row).

    use_two_stage_reduce = (grid_size.x>1 && grid_size.y>1). Must keep grid_size.y=1
    to disable two-stage. Bounding-box check: x_span < grid_size.x, y_span < grid_size.y.
    2 cores at (5,0)-(6,0): x_span=1 < 2 ✓, y_span=0 < 1 ✓.
    """
    k_fused_norm_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0))]  # 2 cores, same row y=0
    )
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 64),  # 128 / 2 = 64 per core
        core_grid=k_fused_norm_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),  # y=1 → use_two_stage_reduce=False
        subblock_w=1,
        block_h=1,
        block_w=2,  # 64/32 = 2 tiles per core
        inplace=False,
    )
    return memcfg, progcfg


def make_q_norm_configs(sub_device_crs):
    """Memory/program configs for Q-norm: [1,1,32,640] on 2 cores (same row y=1).

    2 cores at (5,1)-(6,1): contiguous, same row, both in sub_crs.
    x_span=1 < 2 ✓, y_span=0 < 1 ✓. use_two_stage_reduce=False (y=1).
    block_w = 640/2/32 = 10 tiles per core.  stats_core = (5,1) = sender/bbox-start.
    """
    q_fused_norm_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(6, 1))]  # 2 cores, same row y=1
    )
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 320),  # 640/2 = 320 per core
        core_grid=q_fused_norm_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),  # y=1 → use_two_stage_reduce=False
        subblock_w=2,
        block_h=1,
        block_w=10,  # 320/32 = 10 tiles per core
        inplace=False,
    )
    return memcfg, progcfg


def make_stats_buffer(mesh_device, core):
    """
    Pre-allocate the stats output buffer for fused_rms_minimal (cluster_axis=0).

    The kernel infers num_distributed_devices = stats.padded_shape[-1] / TILE_WIDTH.
    For 8 row devices: shard_shape = (32, 256) → padded_shape[-1] = 256 → num_distributed_devices = 8.

    Use ReplicateTensorToMesh: every device gets an independent L1 copy at the same logical core.
    The kernel on each device writes its gathered stats into this buffer independently.
    """
    total_width = N_ROW_DEVICES * 32  # 8 * 32 = 256
    stats_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, total_width),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    stats_torch = torch.zeros((1, 1, 32, total_width), dtype=torch.bfloat16)
    tt_stats = ttnn.from_torch(
        stats_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=stats_memcfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return tt_stats


def make_weight(mesh_device, weight_torch_1d, per_device_dim):
    """
    Load a 1-D norm weight sharded over cluster_axis=0 (rows).

    weight_torch_1d: [total_dim] e.g. [1024] for K-norm or [5120] for Q-norm
    Reshaped to (1, 1, total_dim//SHARD_HEIGHT, SHARD_HEIGHT) then
    ShardTensor2dMesh(dims=(2, None), mesh_shape=(8,4)) shards dim-2 over rows.
    Each row device gets total_tiles/8 rows = per_device_dim/32 tiles.
    """
    total_dim = weight_torch_1d.shape[0]
    weight_2d = weight_torch_1d.view(1, 1, total_dim // SHARD_HEIGHT, SHARD_HEIGHT)
    return ttnn.as_tensor(
        weight_2d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=CLUSTER_SHAPE),
    )


def make_input_tensor(mesh_device, input_torch, memcfg, per_device_dim):
    """
    Load input tensor (1,1,32,per_device_dim) sharded over cluster_axis=0.
    Each row device holds a different slice of the total norm dim.
    All 4 col devices within a row get the same slice (replicated via None in dim3).
    But actually for decode, each col device has DIFFERENT batch data, and each
    row device has different head/dim data.  However, for this unit test we just
    validate the norm math is correct — so we shard total_data over rows and
    replicate over cols.
    """
    total_dim = per_device_dim * N_ROW_DEVICES
    input_shape_total = (1, 1, 32, total_dim)
    return ttnn.as_tensor(
        input_torch.reshape(input_shape_total),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core test logic
# ──────────────────────────────────────────────────────────────────────────────


def run_olmo_fused_qk_norm(mesh_device, norm_dim_per_device, norm_name):
    """
    Run fused_rms_minimal(cluster_axis=0) for one OLMo QK-norm shape and
    compare against PyTorch reference.

    norm_dim_per_device: 128 (K) or 640 (Q)
    """
    eps = 1e-6
    total_dim = norm_dim_per_device * N_ROW_DEVICES
    input_shape = (1, 1, 32, total_dim)

    # ---- Sub-device setup ----
    # Mirror OLMo's sub_device_crs exactly. dispatch_core_axis=COL means col 0
    # is dispatch, so worker cores start at col 1.
    sub_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 8)),
        }
    )
    worker_sub_device = ttnn.SubDevice([sub_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(42)
    input_torch = torch.randn(input_shape)
    gamma_torch = torch.randn(total_dim)

    # ---- PyTorch reference (normalize over total_dim, per batch row) ----
    input_for_ref = input_torch.reshape(1, 1, 32, total_dim)
    gamma_for_ref = gamma_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,total_dim)
    ref_out = get_torch_rms(input_for_ref, gamma_for_ref, eps)
    logger.info(f"[{norm_name}] ref_out shape: {ref_out.shape}, range [{ref_out.min():.4f}, {ref_out.max():.4f}]")

    # ---- Build TTNN tensors ----
    if norm_dim_per_device == K_DIM_PER_DEVICE:
        memcfg, progcfg = make_k_norm_configs(sub_crs)
        # stats CB is created on sender_core = bbox.start of compute grid = (5,0)
        # The stats tensor MUST be on the same core so set_globally_allocated_address uses the correct L1 address
        stats_core = ttnn.CoreCoord(5, 0)
    else:
        memcfg, progcfg = make_q_norm_configs(sub_crs)
        # Q-norm grid bbox.start = (5,1) (first core of (5,1)-(6,1))
        stats_core = ttnn.CoreCoord(5, 1)

    # Semaphore on the norm grid cores
    semaphore = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)
    tt_stats = make_stats_buffer(mesh_device, stats_core)
    tt_weight = make_weight(mesh_device, gamma_torch, norm_dim_per_device)
    tt_input = make_input_tensor(mesh_device, input_torch, memcfg, norm_dim_per_device)

    logger.info(f"[{norm_name}] stats padded_shape: {tt_stats.shape}, shard_spec: {tt_stats.memory_config()}")

    # ---- fused_rms_minimal(cluster_axis=0) ----
    # stats=tt_stats: num_distributed_devices = stats.padded_shape[-1] / 32
    tt_out = ttnn.fused_rms_minimal(
        tt_input,
        progcfg,
        0,  # cluster_axis = 0 (rows)
        mesh_device,
        semaphore,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=tt_weight,
        stats=tt_stats,
        dtype=ttnn.bfloat16,
        memory_config=memcfg,
    )
    ttnn.synchronize_device(mesh_device)

    # ---- Gather result back to host ----
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE),
    )
    logger.info(f"[{norm_name}] tt_out_torch shape: {tt_out_torch.shape}")
    # Shape: (CLUSTER_SHAPE[1]=4, 1, 32, total_dim) where rows → dim3, cols → dim0
    # Take first "column device's" data (all cols should be identical since input was replicated)
    tt_slice = tt_out_torch[0:1]  # (1, 1, 32, total_dim)
    # Sanity check: look at value range
    finite_mask = torch.isfinite(tt_slice)
    logger.info(f"[{norm_name}] finite elements: {finite_mask.sum().item()}/{tt_slice.numel()}")
    if finite_mask.all():
        logger.info(f"[{norm_name}] value range: [{tt_slice.min():.4f}, {tt_slice.max():.4f}]")
    else:
        logger.warning(f"[{norm_name}] NON-FINITE values found: {(~finite_mask).sum().item()}")

    passing, result = comp_pcc(tt_slice.float(), ref_out.float(), 0.999)
    logger.info(f"[{norm_name}] PCC result: {result}, passing={passing}")

    mesh_device.reset_sub_device_stall_group()

    assert passing, f"[{norm_name}] PCC FAILED: {result}"
    return passing


# ──────────────────────────────────────────────────────────────────────────────
# Pytest entry points
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
def test_olmo_fused_k_norm_pcc(mesh_device, device_params):
    """K-norm: input [1,1,32,128] per device, cluster_axis=0, 8 row devices."""
    run_olmo_fused_qk_norm(mesh_device, K_DIM_PER_DEVICE, "K-norm")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
def test_olmo_fused_q_norm_pcc(mesh_device, device_params):
    """Q-norm: input [1,1,32,640] per device, cluster_axis=0, 8 row devices."""
    run_olmo_fused_qk_norm(mesh_device, Q_DIM_PER_DEVICE, "Q-norm")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Mirror exact model configs — row-8 grid, shared stats core (5,8)
# ──────────────────────────────────────────────────────────────────────────────


def make_k_norm_configs_row8():
    """K-norm at row 8: grid (5,8)-(6,8), stats core (5,8). Matches olmo_model_config.py."""
    k_fused_norm_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 8), ttnn.CoreCoord(6, 8))])
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 64),  # 128 / 2 = 64 per core
        core_grid=k_fused_norm_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),  # y=1 → use_two_stage_reduce=False
        subblock_w=1,
        block_h=1,
        block_w=2,  # 64/32 = 2 tiles per core
        inplace=False,
    )
    return memcfg, progcfg, ttnn.CoreCoord(5, 8)


def make_q_norm_configs_row8():
    """Q-norm at row 8: grid (5,8)-(6,8), stats core (5,8). Matches olmo_model_config.py.
    NOTE: Same physical cores as K-norm — reproducing the model's shared-core allocation."""
    q_fused_norm_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 8), ttnn.CoreCoord(6, 8))])
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 320),  # 640/2 = 320 per core
        core_grid=q_fused_norm_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),  # y=1 → use_two_stage_reduce=False
        subblock_w=1,
        block_h=1,
        block_w=10,  # 320/32=10 tiles per core
        inplace=False,
    )
    return memcfg, progcfg, ttnn.CoreCoord(5, 8)


def run_olmo_fused_qk_norm_row8(mesh_device, norm_dim_per_device, norm_name, make_configs_fn):
    """Run fused_rms_minimal at the row-8 grid matching the model's olmo_model_config.py."""
    eps = 1e-6
    total_dim = norm_dim_per_device * N_ROW_DEVICES
    input_shape = (1, 1, 32, total_dim)

    sub_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 8)),
        }
    )
    worker_sub_device = ttnn.SubDevice([sub_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(42)
    input_torch = torch.randn(input_shape)
    gamma_torch = torch.randn(total_dim)

    input_for_ref = input_torch.reshape(1, 1, 32, total_dim)
    gamma_for_ref = gamma_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    ref_out = get_torch_rms(input_for_ref, gamma_for_ref, eps)

    memcfg, progcfg, stats_core = make_configs_fn()

    semaphore = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)
    tt_stats = make_stats_buffer(mesh_device, stats_core)
    tt_weight = make_weight(mesh_device, gamma_torch, norm_dim_per_device)
    tt_input = make_input_tensor(mesh_device, input_torch, memcfg, norm_dim_per_device)

    logger.info(f"[{norm_name} row8] stats core={stats_core}, stats shape={tt_stats.shape}")

    tt_out = ttnn.fused_rms_minimal(
        tt_input,
        progcfg,
        0,  # cluster_axis=0 (rows)
        mesh_device,
        semaphore,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=tt_weight,
        stats=tt_stats,
        dtype=ttnn.bfloat16,
        memory_config=memcfg,
    )
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE),
    )
    tt_slice = tt_out_torch[0:1]
    finite_mask = torch.isfinite(tt_slice)
    logger.info(f"[{norm_name} row8] finite elements: {finite_mask.sum().item()}/{tt_slice.numel()}")
    if finite_mask.all():
        logger.info(f"[{norm_name} row8] value range: [{tt_slice.min():.4f}, {tt_slice.max():.4f}]")
    else:
        logger.warning(f"[{norm_name} row8] NON-FINITE values: {(~finite_mask).sum().item()}")

    passing, result = comp_pcc(tt_slice.float(), ref_out.float(), 0.999)
    logger.info(f"[{norm_name} row8] PCC result: {result}, passing={passing}")

    mesh_device.reset_sub_device_stall_group()
    assert passing, f"[{norm_name} row8] PCC FAILED: {result}"
    return passing


# ──────────────────────────────────────────────────────────────────────────────
# Axis-1 prototype: shard Q/K norm dimensions over columns instead of rows
# ──────────────────────────────────────────────────────────────────────────────


def make_k_norm_configs_axis1():
    """K-norm prototype for cluster_axis=1: 1024 dims / 4 cols = 256 dims/device."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 8), ttnn.CoreCoord(6, 8))])
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),  # 256 / 2 cores
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        subblock_w=1,
        block_h=1,
        block_w=4,  # 128 / 32
        inplace=False,
    )
    return memcfg, progcfg, ttnn.CoreCoord(5, 8)


def make_q_norm_configs_axis1():
    """Q-norm prototype for cluster_axis=1: 5120 dims / 4 cols = 1280 dims/device."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 8), ttnn.CoreCoord(6, 8))])
    memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 640),  # 1280 / 2 cores
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    progcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        subblock_w=2,
        block_h=1,
        block_w=20,  # 640 / 32
        inplace=False,
    )
    return memcfg, progcfg, ttnn.CoreCoord(5, 8)


def make_stats_buffer_axis1(mesh_device, core):
    total_width = N_COL_DEVICES * 32  # 4 * tile_width = 128
    stats_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, total_width),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    stats_torch = torch.zeros((1, 1, 32, total_width), dtype=torch.bfloat16)
    return ttnn.from_torch(
        stats_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=stats_memcfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def make_weight_axis1(mesh_device, weight_torch_1d):
    total_dim = weight_torch_1d.shape[0]
    weight_2d = weight_torch_1d.view(1, 1, total_dim // SHARD_HEIGHT, SHARD_HEIGHT)
    return ttnn.as_tensor(
        weight_2d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=CLUSTER_SHAPE),
    )


def make_input_tensor_axis1(mesh_device, input_torch, memcfg, norm_dim_per_device):
    total_dim = norm_dim_per_device * N_COL_DEVICES
    return ttnn.as_tensor(
        input_torch.reshape(1, 1, 32, total_dim),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=CLUSTER_SHAPE),
    )


def run_olmo_fused_qk_norm_axis1(mesh_device, total_dim, norm_name, make_configs_fn):
    """Per-op prototype for cluster_axis=1 QK norm across the 4 column devices."""
    eps = 1e-6
    assert total_dim % N_COL_DEVICES == 0
    norm_dim_per_device = total_dim // N_COL_DEVICES

    sub_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 8)),
        }
    )
    worker_sub_device = ttnn.SubDevice([sub_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(123)
    input_torch = torch.randn(1, 1, 32, total_dim)
    gamma_torch = torch.randn(total_dim)
    ref_out = get_torch_rms(input_torch, gamma_torch.view(1, 1, 1, total_dim), eps)

    memcfg, progcfg, stats_core = make_configs_fn()
    semaphore = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)
    stats = make_stats_buffer_axis1(mesh_device, stats_core)
    weight = make_weight_axis1(mesh_device, gamma_torch)
    tt_input = make_input_tensor_axis1(mesh_device, input_torch, memcfg, norm_dim_per_device)

    tt_out = ttnn.fused_rms_minimal(
        tt_input,
        progcfg,
        1,  # cluster_axis=1 (columns)
        mesh_device,
        semaphore,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=weight,
        stats=stats,
        dtype=ttnn.bfloat16,
        memory_config=memcfg,
    )
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=CLUSTER_SHAPE),
    )
    tt_slice = tt_out_torch[0:1]
    finite_mask = torch.isfinite(tt_slice)
    logger.info(f"[{norm_name} axis1] finite elements: {finite_mask.sum().item()}/{tt_slice.numel()}")

    passing, result = comp_pcc(tt_slice.float(), ref_out.float(), 0.999)
    logger.info(f"[{norm_name} axis1] PCC result: {result}, passing={passing}")

    mesh_device.reset_sub_device_stall_group()
    assert passing, f"[{norm_name} axis1] PCC FAILED: {result}"
    return passing


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_fused_k_norm_axis1_prototype(mesh_device, device_params):
    """Axis-1 K-norm prototype: total 1024 dims over 4 column devices."""
    run_olmo_fused_qk_norm_axis1(mesh_device, K_DIM_TOTAL, "K-norm", make_k_norm_configs_axis1)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_fused_q_norm_axis1_prototype(mesh_device, device_params):
    """Axis-1 Q-norm prototype: total 5120 dims over 4 column devices."""
    run_olmo_fused_qk_norm_axis1(mesh_device, Q_DIM_TOTAL, "Q-norm", make_q_norm_configs_axis1)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Sequential K-then-Q with shared stats core (5,8) — model regression
# ──────────────────────────────────────────────────────────────────────────────


def run_olmo_fused_qk_sequential(mesh_device):
    """Run K-norm then Q-norm back-to-back, both sharing stats core (5,8).
    This mirrors the exact model decode trace: same persistent stats buffer address
    reused for sequential fused_rms_minimal calls.
    Hypothesis: if Q reads stale K stats from the shared core, PCC fails.
    """
    eps = 1e-6

    sub_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 8)),
        }
    )
    worker_sub_device = ttnn.SubDevice([sub_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(99)

    # K inputs
    k_total = K_DIM_PER_DEVICE * N_ROW_DEVICES
    k_input_torch = torch.randn(1, 1, 32, k_total)
    k_gamma_torch = torch.randn(k_total)
    k_ref = get_torch_rms(k_input_torch, k_gamma_torch.view(1, 1, 1, k_total), eps)

    # Q inputs
    q_total = Q_DIM_PER_DEVICE * N_ROW_DEVICES
    q_input_torch = torch.randn(1, 1, 32, q_total)
    q_gamma_torch = torch.randn(q_total)
    q_ref = get_torch_rms(q_input_torch, q_gamma_torch.view(1, 1, 1, q_total), eps)

    # Build configs (row 8, matching model)
    k_memcfg, k_progcfg, k_stats_core = make_k_norm_configs_row8()
    q_memcfg, q_progcfg, q_stats_core = make_q_norm_configs_row8()
    # Both stats cores are (5,8) — same as the model's LAYERNORM_QK_K_STATS and LAYERNORM_QK_Q_STATS
    assert (
        k_stats_core.x == q_stats_core.x and k_stats_core.y == q_stats_core.y
    ), "Stats cores must be identical to reproduce the model's shared-core bug"

    # Two separate stat buffers but SAME physical core — this is exactly the model state
    k_stats_buf = make_stats_buffer(mesh_device, k_stats_core)
    q_stats_buf = make_stats_buffer(mesh_device, q_stats_core)

    # Shared semaphore (same as the model's gather_semaphore_handles[0])
    semaphore = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)

    k_weight = make_weight(mesh_device, k_gamma_torch, K_DIM_PER_DEVICE)
    q_weight = make_weight(mesh_device, q_gamma_torch, Q_DIM_PER_DEVICE)
    k_input = make_input_tensor(mesh_device, k_input_torch, k_memcfg, K_DIM_PER_DEVICE)
    q_input = make_input_tensor(mesh_device, q_input_torch, q_memcfg, Q_DIM_PER_DEVICE)

    logger.info(f"[sequential] K stats core={k_stats_core}, Q stats core={q_stats_core} (identical)")

    # K first
    k_out = ttnn.fused_rms_minimal(
        k_input,
        k_progcfg,
        0,
        mesh_device,
        semaphore,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=k_weight,
        stats=k_stats_buf,
        dtype=ttnn.bfloat16,
        memory_config=k_memcfg,
    )
    ttnn.synchronize_device(mesh_device)

    # Q second (same semaphore, same stats core)
    q_out = ttnn.fused_rms_minimal(
        q_input,
        q_progcfg,
        0,
        mesh_device,
        semaphore,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=q_weight,
        stats=q_stats_buf,
        dtype=ttnn.bfloat16,
        memory_config=q_memcfg,
    )
    ttnn.synchronize_device(mesh_device)

    # Check K output
    k_out_torch = ttnn.to_torch(
        k_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE)
    )[0:1]
    k_passing, k_result = comp_pcc(k_out_torch.float(), k_ref.float(), 0.999)
    logger.info(f"[sequential] K PCC: {k_result}, passing={k_passing}")

    # Check Q output
    q_out_torch = ttnn.to_torch(
        q_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE)
    )[0:1]
    q_passing, q_result = comp_pcc(q_out_torch.float(), q_ref.float(), 0.999)
    logger.info(f"[sequential] Q PCC: {q_result}, passing={q_passing}")

    mesh_device.reset_sub_device_stall_group()
    assert k_passing, f"[sequential] K PCC FAILED: {k_result}"
    assert q_passing, f"[sequential] Q PCC FAILED: {q_result}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_fused_k_norm_row8(mesh_device, device_params):
    """K-norm at row-8 grid (5,8)-(6,8) — matching olmo_model_config.py exactly."""
    run_olmo_fused_qk_norm_row8(mesh_device, K_DIM_PER_DEVICE, "K-norm", make_k_norm_configs_row8)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_fused_q_norm_row8(mesh_device, device_params):
    """Q-norm at row-8 grid (5,8)-(6,8) — matching olmo_model_config.py exactly.
    NOTE: same physical cores as K-norm row8 test."""
    run_olmo_fused_qk_norm_row8(mesh_device, Q_DIM_PER_DEVICE, "Q-norm", make_q_norm_configs_row8)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_fused_qk_norm_sequential(mesh_device, device_params):
    """Sequential K-then-Q with shared stats core (5,8) — model-regression test.
    Reproduces the aliasing scenario: LAYERNORM_QK_K_STATS and LAYERNORM_QK_Q_STATS
    both pinned to core (5,8), called back-to-back in the same trace."""
    run_olmo_fused_qk_sequential(mesh_device)


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Traced regression — K-then-Q in a metal trace (decode-like)
# ──────────────────────────────────────────────────────────────────────────────


def run_olmo_fused_qk_traced(mesh_device):
    """Capture K-norm + Q-norm (row-8 grid) in a metal trace and replay it.

    This mirrors the decode forward pass: the trace captures both ops, then
    execute_trace replays them.  We also use TWO distinct semaphores (semaphore[0]
    for K, semaphore[1] for Q), matching TT_CCL's double-buffering pattern.
    If the output PCCs fail only in traced mode (not in eager mode), the bug
    is trace-specific (semaphore reset, CB aliasing in replay, etc.).

    IMPORTANT: Use the FULL device core range (0,0)-(6,8) as the sub-device,
    matching what llama_model.py setup_decode() uses. A restricted sub-device
    causes trace hangs because the ETH/fabric synchronisation cores are excluded.
    """
    eps = 1e-6

    # Full core range — matches model's setup_decode() sub-device
    sub_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 8))])
    worker_sub_device = ttnn.SubDevice([sub_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(77)

    k_total = K_DIM_PER_DEVICE * N_ROW_DEVICES
    q_total = Q_DIM_PER_DEVICE * N_ROW_DEVICES
    k_input_torch = torch.randn(1, 1, 32, k_total)
    q_input_torch = torch.randn(1, 1, 32, q_total)
    k_gamma_torch = torch.randn(k_total)
    q_gamma_torch = torch.randn(q_total)

    k_ref = get_torch_rms(k_input_torch, k_gamma_torch.view(1, 1, 1, k_total), eps)
    q_ref = get_torch_rms(q_input_torch, q_gamma_torch.view(1, 1, 1, q_total), eps)

    k_memcfg, k_progcfg, k_stats_core = make_k_norm_configs_row8()
    q_memcfg, q_progcfg, q_stats_core = make_q_norm_configs_row8()

    # Two semaphores — double-buffered like TT_CCL: K uses [0], Q uses [1]
    semaphore_k = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)
    semaphore_q = ttnn.create_global_semaphore(mesh_device, sub_crs, 0)

    k_stats_buf = make_stats_buffer(mesh_device, k_stats_core)
    q_stats_buf = make_stats_buffer(mesh_device, q_stats_core)

    k_weight = make_weight(mesh_device, k_gamma_torch, K_DIM_PER_DEVICE)
    q_weight = make_weight(mesh_device, q_gamma_torch, Q_DIM_PER_DEVICE)

    # Persistent output tensors — needed for trace (ops can't allocate new buffers during replay)
    k_out_buf = ttnn.from_torch(
        torch.zeros(1, 1, 32, k_total),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=k_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
    )
    q_out_buf = ttnn.from_torch(
        torch.zeros(1, 1, 32, q_total),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=q_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=CLUSTER_SHAPE),
    )

    k_input = make_input_tensor(mesh_device, k_input_torch, k_memcfg, K_DIM_PER_DEVICE)
    q_input = make_input_tensor(mesh_device, q_input_torch, q_memcfg, Q_DIM_PER_DEVICE)

    logger.info("[traced] Starting trace capture of K-norm + Q-norm")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    k_out = ttnn.fused_rms_minimal(
        k_input,
        k_progcfg,
        0,
        mesh_device,
        semaphore_k,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=k_weight,
        stats=k_stats_buf,
        dtype=ttnn.bfloat16,
        memory_config=k_memcfg,
    )
    q_out = ttnn.fused_rms_minimal(
        q_input,
        q_progcfg,
        0,
        mesh_device,
        semaphore_q,
        topology=ttnn.Topology.Linear,
        num_links=1,
        epsilon=eps,
        weight=q_weight,
        stats=q_stats_buf,
        dtype=ttnn.bfloat16,
        memory_config=q_memcfg,
    )

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info("[traced] Trace captured. Executing trace...")

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    logger.info("[traced] Trace executed.")

    k_out_torch = ttnn.to_torch(
        k_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE)
    )[0:1]
    q_out_torch = ttnn.to_torch(
        q_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=CLUSTER_SHAPE)
    )[0:1]

    k_finite = torch.isfinite(k_out_torch)
    q_finite = torch.isfinite(q_out_torch)
    logger.info(
        f"[traced] K finite: {k_finite.sum()}/{k_out_torch.numel()}, Q finite: {q_finite.sum()}/{q_out_torch.numel()}"
    )

    k_passing, k_result = comp_pcc(k_out_torch.float(), k_ref.float(), 0.999)
    q_passing, q_result = comp_pcc(q_out_torch.float(), q_ref.float(), 0.999)
    logger.info(f"[traced] K PCC: {k_result}, passing={k_passing}")
    logger.info(f"[traced] Q PCC: {q_result}, passing={q_passing}")

    ttnn.release_trace(mesh_device, trace_id)
    mesh_device.reset_sub_device_stall_group()

    assert k_passing, f"[traced] K PCC FAILED: {k_result}"
    assert q_passing, f"[traced] Q PCC FAILED: {q_result}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.skip(
    reason="Standalone QK-norm unit coverage is eager/sequential; trace is covered by decode/prefill integration."
)
def test_olmo_fused_qk_norm_traced(mesh_device, device_params):
    """K-norm + Q-norm captured in a metal trace and replayed — decode regression.
    Uses two distinct semaphores (double-buffered) and persistent output tensors,
    mirroring the actual OLMo decode trace setup."""
    run_olmo_fused_qk_traced(mesh_device)
