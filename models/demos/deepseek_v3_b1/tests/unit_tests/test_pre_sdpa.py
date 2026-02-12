# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN PreSDPA Test
Tests pre-SDPA fused operation with full pipeline:
- CCL Broadcast -> RMSNorm -> Matmul -> Gather -> RMSNorm2 -> Matmul2 (shuffled) -> Matmul3 (Qnope only) & RoPE (Qrope only) -> Interleaved Pre-SDPA Output
- Qnope output: [64, 1, 512] after matmul3
- Qrope output: [64, 1, 64] after RoPE
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA
from models.demos.deepseek_v3_b1.utils import shuffle_weights_for_interleaved_qnope_qrope


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("secondary_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2), (1, 1)])
@pytest.mark.parametrize("num_iters", [(30)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_pre_sdpa(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    cluster_axis,
    secondary_cluster_axis,
    num_iters,
):
    """Test TTNN pre-SDPA fused operation with CCL broadcast and full Qnope/Qrope pipeline"""

    num_devices = mesh_rows * mesh_cols
    skip_ccl = False
    if num_devices == 1:
        skip_ccl = True

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh used by the test
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Configure a single worker sub-device covering the full compute grid
    device_grid_size = submesh.compute_with_storage_grid_size()

    # ========================================================================
    # Configuration
    # ========================================================================
    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    # Head configuration
    NUM_QNOPE_HEADS = 64
    NUM_QROPE_HEADS = 64
    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    HEADS_PER_ROW = 8
    QNOPE_OUT_DIM = 512  # Output dimension per Qnope head after matmul3

    KNOPE_DIM = 512
    KROPE_DIM = 64

    # Qnope/Qrope grid configuration (must match head configuration)
    QNOPE_GRID_COLS = 8  # 8 Qnope cores per row (1 head each)
    QROPE_GRID_COLS = 4  # 4 Qrope cores per row (2 heads each)
    TOTAL_COLS = QNOPE_GRID_COLS + QROPE_GRID_COLS  # 12 columns required

    # Skip test on P100 devices (which have 11 columns instead of 12)
    if device_grid_size.x < TOTAL_COLS:
        pytest.skip(
            f"Device grid {device_grid_size.x}x{device_grid_size.y} too small for {TOTAL_COLS} columns required (P100 has 11 columns)"
        )

    matmul2_grid_x = min(TOTAL_COLS, device_grid_size.x)  # Must be exactly 12 for correct sharding
    matmul2_grid_y = 8

    # Matmul2 output width (interleaved Qnope/Qrope)
    matmul2_width = NUM_QNOPE_HEADS * QNOPE_HEAD_DIM + NUM_QROPE_HEADS * QROPE_HEAD_DIM  # 8192 + 4096 = 12288
    matmul2_weights_shape = (1536, matmul2_width)
    qnope_num_cores = QNOPE_GRID_COLS * matmul2_grid_y  # 64 cores
    qrope_num_cores = QROPE_GRID_COLS * matmul2_grid_y  # 32 cores

    # Mcast/gather core coordinates (same as RMSNorm input core)
    # Use the full device grid width for mcast core (last column)
    mcast_core_x = device_grid_size.x - 1  # Last column
    mcast_core_y = 9

    # KV Cache Branch grid configuration
    kv_cache_branch_start_offset = (0, 8)  # Offset for the operation grid
    kv_cache_branch_grid = (9, 2)  # Grid dimensions for the operation

    kv_cache_branch_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
                ttnn.CoreCoord(
                    kv_cache_branch_grid[0] + kv_cache_branch_start_offset[0] - 1,
                    kv_cache_branch_grid[1] + kv_cache_branch_start_offset[1] - 1,
                ),
            )
        }
    )
    kv_cache_branch_rms_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
                ttnn.CoreCoord(kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
            )
        }
    )
    kv_cache_branch_rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], 1 + kv_cache_branch_start_offset[1]),
            )
        }
    )
    dkv_matmul_weights_shape = (7168, KNOPE_DIM + KROPE_DIM)

    tile = ttnn.Tile([1, 32])

    # RMSNorm2 parameters (1536 elements = 3 tiles of 16x32)
    rmsnorm2_width = 1536

    logger.info(f"Device grid: {device_grid_size.x}x{device_grid_size.y}")
    logger.info(f"Qnope cores: {qnope_num_cores}, Qrope cores: {qrope_num_cores}")

    # ========================================================================
    # Create PyTorch tensors
    # ========================================================================
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)
    torch_matmul_weights = torch.randn(matmul_weights_shape, dtype=torch.bfloat16)
    torch_rmsnorm2_gamma = torch.randn((1, rmsnorm2_width), dtype=torch.bfloat16)

    # Matmul2 weights - create unshuffled first, then shuffle for device
    torch_matmul2_weights_unshuffled = torch.randn(matmul2_weights_shape, dtype=torch.bfloat16)
    torch_matmul2_weights_shuffled = shuffle_weights_for_interleaved_qnope_qrope(
        torch_matmul2_weights_unshuffled,
        num_qnope_heads=NUM_QNOPE_HEADS,
        num_qrope_heads=NUM_QROPE_HEADS,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
    )

    # Matmul3 weights - [num_qnope_heads, qnope_head_dim, qnope_out_dim] for golden
    # but [qnope_head_dim, qnope_out_dim] per core for device (height sharded)
    torch_matmul3_weights = torch.randn((NUM_QNOPE_HEADS, QNOPE_HEAD_DIM, QNOPE_OUT_DIM), dtype=torch.bfloat16)

    # ========================================================================
    # Create RoPE tensors (sin, cos, trans_mat)
    # ========================================================================
    max_seq_len = 8192
    position_id = 0  # Decode mode: first token
    position_ids = torch.tensor([position_id])  # [batch]

    # Create cos/sin matrices in Meta-style format
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, QROPE_HEAD_DIM, 2, dtype=torch.float32) / QROPE_HEAD_DIM))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Meta-style: stack [cos(t), cos(t)] interleaved
    torch_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)  # [max_seq_len, qrope_head_dim]
    torch_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)  # [max_seq_len, qrope_head_dim]

    # Transformation matrix for RoPE
    torch_trans_mat = get_rot_transformation_mat()  # [1, 1, 32, 32]

    # ========================================================================
    # Create TTNN tensors
    # ========================================================================

    # Shard spec: single core for input, gamma (on mcast/gather core)
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create mesh tensors for input and intermediate (CCL broadcast destination)
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    device_tensors = []
    intermediate_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if skip_ccl:
                # Single-device mode: all devices have the input
                device_tensors.append(torch_input)  # (1, 7168)
            elif row == sender_row and col == sender_col:
                # Only sender device has actual input data
                device_tensors.append(torch_input)  # (1, 7168)
            else:
                # All other devices start with zeros
                device_tensors.append(torch.zeros_like(torch_input))  # (1, 7168)
            intermediate_tensors.append(torch.zeros_like(torch_input))

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    intermediate_mesh_tensor_torch = torch.cat(intermediate_tensors, dim=0)

    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    intermediate_tensor_mesh = ttnn.from_torch(
        intermediate_mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Matmul weights tensor - width sharded on 6x8 grid (48 cores)
    matmul_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))
    num_matmul_cores = 6 * 8
    matmul_shard_shape = (matmul_weights_shape[0], matmul_weights_shape[1] // num_matmul_cores)
    matmul_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul_grid}),
        matmul_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul_shard_spec)

    ttnn_matmul_weights = ttnn.from_torch(
        torch_matmul_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Matmul2 weights tensor (shuffled) - width sharded on 8x12 grid
    matmul2_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(matmul2_grid_x - 1, matmul2_grid_y - 1))
    matmul2_num_cores = matmul2_grid_x * matmul2_grid_y
    matmul2_shard_shape = (matmul2_weights_shape[0], matmul2_weights_shape[1] // matmul2_num_cores)
    matmul2_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul2_grid}),
        matmul2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul2_shard_spec
    )

    ttnn_matmul2_weights = ttnn.from_torch(
        torch_matmul2_weights_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # RMSNorm2 gamma tensor
    rmsnorm2_gamma_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        (1, rmsnorm2_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    rmsnorm2_gamma_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, rmsnorm2_gamma_shard_spec
    )
    ttnn_rmsnorm2_gamma = ttnn.from_torch(
        torch_rmsnorm2_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=rmsnorm2_gamma_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Matmul3 weights tensor - height sharded on Qnope grid (64 cores)
    # Each core gets [128, 512] = shape per core
    qnope_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(QNOPE_GRID_COLS - 1, matmul2_grid_y - 1))
    # Flatten matmul3 weights for height sharding: [num_heads * K, N] = [64 * 128, 512] = [8192, 512]
    torch_matmul3_weights_flat = torch_matmul3_weights.reshape(NUM_QNOPE_HEADS * QNOPE_HEAD_DIM, QNOPE_OUT_DIM)
    matmul3_shard_shape = (QNOPE_HEAD_DIM, QNOPE_OUT_DIM)  # [128, 512] per core
    matmul3_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qnope_grid}),
        matmul3_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul3_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, matmul3_shard_spec
    )

    ttnn_matmul3_weights = ttnn.from_torch(
        torch_matmul3_weights_flat,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul3_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # SDPA input tensor - height sharded on SDPA input grid (cols 0-3, rows 1-2)
    # Each SDPA Input core receives 8 interleaved heads: 8 × (512 + 64) = 8 × 576 = 4608 elements
    # SDPA Input grid: 4×2 rectangle at logical (0,1)-(3,2)
    SDPA_INPUT_GRID_COLS = 4
    SDPA_INPUT_GRID_ROWS = 2
    SDPA_INPUT_NUM_CORES = SDPA_INPUT_GRID_COLS * SDPA_INPUT_GRID_ROWS  # 8 cores
    COMBINED_HEAD_SIZE = QNOPE_OUT_DIM + QROPE_HEAD_DIM  # 512 + 64 = 576
    SDPA_INPUT_ELEMENTS_PER_CORE = HEADS_PER_ROW * COMBINED_HEAD_SIZE  # 8 * 576 = 4608

    sdpa_input_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 1), ttnn.CoreCoord(SDPA_INPUT_GRID_COLS - 1, 1 + SDPA_INPUT_GRID_ROWS - 1)
    )
    sdpa_input_output_shape = (SDPA_INPUT_NUM_CORES, SDPA_INPUT_ELEMENTS_PER_CORE)  # [8, 4608] total
    sdpa_input_output_shard_shape = (1, SDPA_INPUT_ELEMENTS_PER_CORE)  # [1, 4608] per core
    sdpa_input_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({sdpa_input_grid}),
        sdpa_input_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_input_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_input_output_shard_spec
    )
    torch_sdpa_input_output = torch.zeros(sdpa_input_output_shape, dtype=torch.bfloat16)
    ttnn_sdpa_input_output = ttnn.from_torch(
        torch_sdpa_input_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=sdpa_input_output_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # RoPE tensors - sharded on Qrope grid (4x8 = 32 cores)
    qrope_grid = ttnn.CoreRange(
        ttnn.CoreCoord(QNOPE_GRID_COLS, 0), ttnn.CoreCoord(QNOPE_GRID_COLS + QROPE_GRID_COLS - 1, matmul2_grid_y - 1)
    )

    # For decode mode, cos/sin are indexed by position: [1, batch, 1, head_dim]
    # Shape: [1, 1, 1, qrope_head_dim] - broadcast multiply will use row 0

    # Cos/Sin sharding: HEIGHT_SHARDED on qrope grid
    # Each core gets full head_dim (since cos/sin are reused for all heads on that core)
    # Shape per core: (1, qrope_head_dim) = (1, 64)
    # Shared with KV Cache branch, double check if modifying these tensors
    cos_selected = torch_cos[position_ids].unsqueeze(0).unsqueeze(2)  # [1, batch, 1, qrope_head_dim] = [1, 1, 1, 64]
    sin_selected = torch_sin[position_ids].unsqueeze(0).unsqueeze(2)  # [1, batch, 1, qrope_head_dim] = [1, 1, 1, 64]

    qrope_cos_sin_shard_shape = (1, QROPE_HEAD_DIM)  # [1, 64] per core
    qrope_cos_sin_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qrope_grid}),
        qrope_cos_sin_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    qrope_cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, qrope_cos_sin_shard_spec
    )

    # Repeat cos/sin for all qrope cores: [1, 1, num_cores, qrope_head_dim]
    # Each core gets the same cos/sin (reused for its 2 heads)
    cos_replicated = cos_selected.repeat(1, 1, qrope_num_cores, 1)  # [1, 1, 32, 64]
    sin_replicated = sin_selected.repeat(1, 1, qrope_num_cores, 1)  # [1, 1, 32, 64]

    ttnn_cos = ttnn.from_torch(
        cos_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_cos_sin_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    ttnn_sin = ttnn.from_torch(
        sin_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_cos_sin_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Trans_mat: [1, 1, 32, 32] - repeat for all qrope cores
    # Each core gets full [32, 32] transformation matrix (reused for all heads)
    trans_mat_replicated = torch_trans_mat.repeat(
        1, 1, qrope_num_cores + kv_cache_branch_rope_crs.num_cores(), 1
    )  # [1, 1, 32, 32]
    trans_mat_crs = kv_cache_branch_rope_crs.merge(ttnn.CoreRangeSet({qrope_grid}))
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    trans_shard_shape = (ttnn.TILE_SIZE, ttnn.TILE_SIZE)  # [32, 32] per core
    trans_shard_spec = ttnn.ShardSpec(
        trans_mat_crs,
        trans_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    ttnn_trans_mat = ttnn.from_torch(
        trans_mat_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=trans_mem_config,
        tile=trans_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # KV Cache Branch
    torch_dkv_matmul_weights = torch.randn(dkv_matmul_weights_shape, dtype=torch.bfloat16)
    num_shards = kv_cache_branch_crs.num_cores()
    shard_width = dkv_matmul_weights_shape[1] // num_shards
    new_shard_order = [0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17]
    torch_dkv_matmul_weights_shards = torch_dkv_matmul_weights.reshape(
        dkv_matmul_weights_shape[0], num_shards, shard_width
    )
    torch_dkv_matmul_weights_shuffled = torch_dkv_matmul_weights_shards[:, new_shard_order, :].reshape(
        dkv_matmul_weights_shape[0], dkv_matmul_weights_shape[1]
    )
    dkv_matmul_weights_shard_spec = ttnn.ShardSpec(
        kv_cache_branch_crs,
        (dkv_matmul_weights_shape[0], shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    dkv_matmul_weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dkv_matmul_weights_shard_spec
    )
    ttnn_dkv_matmul_weights = ttnn.from_torch(
        torch_dkv_matmul_weights_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=dkv_matmul_weights_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    torch_dkv_rmsnorm_gamma = torch.randn((1, KNOPE_DIM), dtype=torch.bfloat16)
    dkv_rmsnorm_gamma_shard_spec = ttnn.ShardSpec(
        kv_cache_branch_rms_crs,
        (1, KNOPE_DIM),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dkv_rmsnorm_gamma_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, dkv_rmsnorm_gamma_shard_spec
    )

    ttnn_dkv_rmsnorm_gamma = ttnn.from_torch(
        torch_dkv_rmsnorm_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=dkv_rmsnorm_gamma_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    krope_num_heads = 1
    krope_cos_sin_shard_spec = ttnn.ShardSpec(
        kv_cache_branch_rope_crs,
        (krope_num_heads, KROPE_DIM // 2),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    krope_cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, krope_cos_sin_shard_spec
    )

    ttnn_krope_cos = ttnn.from_torch(
        cos_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=krope_cos_sin_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_krope_sin = ttnn.from_torch(
        sin_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=krope_cos_sin_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    out_ready_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore]

    # ========================================================================
    # Run pre-SDPA operation
    # ========================================================================
    logger.info("Running pre-SDPA operation...")

    for i in range(num_iters):
        ttnn_sdpa_input_result = PreSDPA.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_matmul_weights,
            ttnn_rmsnorm2_gamma,
            ttnn_matmul2_weights,
            ttnn_matmul3_weights,
            ttnn_sin,
            ttnn_cos,
            ttnn_trans_mat,
            ttnn_krope_cos,
            ttnn_krope_sin,
            ttnn_dkv_matmul_weights,
            ttnn_dkv_rmsnorm_gamma,
            ttnn_sdpa_input_output,
            sender_coord,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            secondary_cluster_axis=secondary_cluster_axis,
            epsilon=epsilon,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=skip_ccl,
        )
    ttnn.synchronize_device(submesh)

    # Convert back to torch for verification
    sdpa_input_output_torch = ttnn.to_torch(
        ttnn_sdpa_input_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    # Golden uses shuffled weights to produce same interleaved output
    _, _, torch_sdpa_expected, torch_kv_cache_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights_shuffled,  # Use shuffled weights
        torch_matmul3_weights,
        torch_sin,
        torch_cos,
        position_ids,
        torch_dkv_matmul_weights,
        torch_dkv_rmsnorm_gamma,
        epsilon=epsilon,
        num_qnope_heads=NUM_QNOPE_HEADS,
        num_qrope_heads=NUM_QROPE_HEADS,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        knope_dim=KNOPE_DIM,
        krope_dim=KROPE_DIM,
    )

    slice_size = sdpa_input_output_shape[0]
    expected_width = 4608

    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * slice_size
        end = start + slice_size
        # Trim to expected width for comparison
        received = sdpa_input_output_torch[start:end, :expected_width]

        # Golden SDPA Input shape: [8, 8, 576] -> reshape to [8, 4608] to match device output
        # The 4×2 grid with ROW_MAJOR orientation gives indices 0-7 matching source rows 0-7
        # Each combined head is (qnope[512], qrope[64]) where qrope has been processed by RoPE
        torch_sdpa_input_expected_flat = torch_sdpa_expected.reshape(SDPA_INPUT_NUM_CORES, SDPA_INPUT_ELEMENTS_PER_CORE)
        if received.shape != torch_sdpa_input_expected_flat.shape:
            logger.error(
                f"Shape mismatch at device {device_idx}: got {received.shape}, expected {torch_sdpa_expected.shape}"
            )
            continue

        max_diff = torch.max(torch.abs(received - torch_sdpa_input_expected_flat)).item()
        mean_diff = torch.mean(torch.abs(received - torch_sdpa_input_expected_flat)).item()

        logger.info(f"Device {device_idx}: Max diff={max_diff}, Mean diff={mean_diff}")

        passing, pcc_message = comp_pcc(torch_sdpa_input_expected_flat, received, 0.99)
        logger.info(f"Device {device_idx}: {pcc_message}")

        assert passing, f"Device {device_idx} failed: {pcc_message}"

    logger.info("✓ PreSDPA mesh test passed!")

    # Clean up trace and sub-device state before fixture teardown
    # This ensures profiler data is properly flushed before close_mesh_device
    ttnn.synchronize_device(submesh)
