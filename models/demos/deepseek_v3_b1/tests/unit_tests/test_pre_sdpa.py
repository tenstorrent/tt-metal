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
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
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
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("position_id", [0, 2130])
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
    position_id,
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

    assert QROPE_HEAD_DIM == KROPE_DIM, "Qrope and Krope head dimensions must match"

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
    matmul_h, matmul_w = matmul_weights_shape
    # Pack [H, W] -> [H/2, 2W] by pairing each row i from K-half0 and K-half1 as [half0_i | half1_i].
    torch_matmul_weights_packed = (
        torch_matmul_weights.reshape(2, matmul_h // 2, matmul_w).permute(1, 0, 2).reshape(matmul_h // 2, 2 * matmul_w)
    )
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

    # Matmul1 packed weights tensor
    # Pack [7168, 1536] -> [3584, 3072] by concatenating K-halves across width.
    # Width-shard over full 96-core grid, giving per-core shard [3584, 32].
    matmul_grid_x = 12
    matmul_grid_y = 8
    matmul_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(matmul_grid_x - 1, matmul_grid_y - 1))
    num_matmul_cores = matmul_grid_x * matmul_grid_y
    # WIDTH_SHARDED requires shard height == tensor height.
    # Packed tensor is [3584, 3072], so per-core shard is [3584, 32].
    matmul_packed_shard_shape = (
        matmul_weights_shape[0] // 2,
        (matmul_weights_shape[1] * 2) // num_matmul_cores,
    )
    matmul_packed_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul_grid}),
        matmul_packed_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul_packed_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul_packed_shard_spec
    )

    ttnn_matmul_weights = ttnn.from_torch(
        torch_matmul_weights_packed,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=matmul_packed_mem_config,
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
    # After 3-phase CreateQHeads tilization:
    #   Each SDPA Input core receives 8 heads, tilized as [8, 576] with [8, 32] tiles
    #   Total: 8 cores × 8 heads = 64 rows, 576 elements per head
    # SDPA Input grid: 4×2 rectangle at logical (0,1)-(3,2)
    SDPA_INPUT_GRID_COLS = 4
    SDPA_INPUT_GRID_ROWS = 2
    SDPA_INPUT_NUM_CORES = SDPA_INPUT_GRID_COLS * SDPA_INPUT_GRID_ROWS  # 8 cores
    COMBINED_HEAD_SIZE = QNOPE_OUT_DIM + QROPE_HEAD_DIM  # 512 + 64 = 576

    sdpa_input_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 1), ttnn.CoreCoord(SDPA_INPUT_GRID_COLS - 1, 1 + SDPA_INPUT_GRID_ROWS - 1)
    )
    sdpa_tile = ttnn.Tile([8, 32])  # Tilize tile shape for CreateQHeads output
    sdpa_input_output_shape = (SDPA_INPUT_NUM_CORES * HEADS_PER_ROW, COMBINED_HEAD_SIZE)  # [64, 576] total
    sdpa_input_output_shard_shape = (HEADS_PER_ROW, COMBINED_HEAD_SIZE)  # [8, 576] per core
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
        tile=sdpa_tile,
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

    ttnn_qrope_cos = ttnn.from_torch(
        cos_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_cos_sin_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    ttnn_qrope_sin = ttnn.from_torch(
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
    # new_shard_order = [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8, 16, 17]
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

    # KV Cache tensor in DRAM sharded
    # Create KV cache (non-paged) based on max seq len
    program_config = FlashMLADecode.ProgramConfig(
        k_chunk_size=128,
        exp_approx_mode=False,  # Use exact exp for higher precision
    )
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    kvpe_dim = KNOPE_DIM + KROPE_DIM
    cache_shape = (1, 1, max_seq_len, kvpe_dim)
    torch_kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    # ND sharding with ROUND_ROBIN_1D distribution across DRAM banks
    # Each shard = one k_chunk (k_chunk_size x kvpe_dim), distributed round-robin
    # Use optimal DRAM bank order matching S block work assignment for locality
    grid = program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )
    num_chunks = max_seq_len // program_config.k_chunk_size
    num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
    logger.info(
        f"KV cache: ND sharded, DRAM banks: {num_banks} (optimal order: {grid.OPTIMAL_DRAM_BANK_ORDER}), chunks: {num_chunks}, shard_shape: [1, 1, {program_config.k_chunk_size}, {kvpe_dim}]"
    )

    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

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
            ttnn_qrope_sin,
            ttnn_qrope_cos,
            ttnn_trans_mat,
            ttnn_krope_cos,
            ttnn_krope_sin,
            ttnn_dkv_matmul_weights,
            ttnn_dkv_rmsnorm_gamma,
            ttnn_kv_cache,
            position_ids,
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

    kv_cache_output_torch = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    # Convert back to torch for verification
    sdpa_input_output_torch = ttnn.to_torch(
        ttnn_sdpa_input_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    # Golden uses unshuffled weights (sequential output: all QNOPE, then all QROPE)
    _, _, torch_sdpa_expected, torch_kv_cache_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights_unshuffled,  # Use unshuffled weights
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
    expected_width = COMBINED_HEAD_SIZE  # 576

    # KV Cache is same across devices in 4x2 submesh
    expected_nope = torch_kv_cache_expected[..., :KNOPE_DIM]
    expected_rope = torch_kv_cache_expected[..., KNOPE_DIM:]

    for device_idx in range(mesh_rows * mesh_cols):
        start = device_idx * slice_size
        end = start + slice_size
        # Trim to expected width for comparison
        received = sdpa_input_output_torch[start:end, :expected_width]

        # Golden SDPA Input shape: [64, 576] (already reshaped by golden function)
        # Each row is one head: [qnope[512], qrope[64]]
        # 8 cores × 8 heads = 64 rows
        torch_sdpa_input_expected_flat = torch_sdpa_expected
        if received.shape != torch_sdpa_input_expected_flat.shape:
            logger.error(
                f"Shape mismatch at device {device_idx}: got {received.shape}, expected {torch_sdpa_input_expected_flat.shape}"
            )
            continue

        max_diff = torch.max(torch.abs(received - torch_sdpa_input_expected_flat)).item()
        mean_diff = torch.mean(torch.abs(received - torch_sdpa_input_expected_flat)).item()

        logger.info(f"Device {device_idx}: Max diff={max_diff}, Mean diff={mean_diff}")

        passing, pcc_message = comp_pcc(torch_sdpa_input_expected_flat, received, 0.99)
        logger.info(f"Device {device_idx}: {pcc_message}")

        assert (
            passing
        ), f"Device {device_idx} failed: {pcc_message}"  # Read back from kv cache tensor in DRAM to check PCC

        compare_kv_cache = kv_cache_output_torch[..., position_id, :]

        # Split into nope (first 512 elements) and rope (last 64 elements)
        compare_nope = compare_kv_cache[..., :KNOPE_DIM]
        compare_rope = compare_kv_cache[..., KNOPE_DIM:]

        # Check nope portion
        nope_max_diff = torch.max(torch.abs(expected_nope - compare_nope)).item()
        nope_mean_diff = torch.mean(torch.abs(expected_nope - compare_nope)).item()
        logger.info(f"KV Cache NOPE absolute difference: {nope_max_diff}")
        logger.info(f"KV Cache NOPE mean absolute difference: {nope_mean_diff}")
        nope_passing, nope_pcc_message = comp_pcc(compare_nope, expected_nope, 0.98)
        logger.info(f"KV Cache NOPE PCC: {nope_pcc_message}")

        # Check rope portion
        rope_max_diff = torch.max(torch.abs(expected_rope - compare_rope)).item()
        rope_mean_diff = torch.mean(torch.abs(expected_rope - compare_rope)).item()
        logger.info(f"KV Cache ROPE absolute difference: {rope_max_diff}")
        logger.info(f"KV Cache ROPE mean absolute difference: {rope_mean_diff}")
        rope_passing, rope_pcc_message = comp_pcc(compare_rope, expected_rope, 0.98)
        logger.info(f"KV Cache ROPE PCC: {rope_pcc_message}")

        assert nope_passing, f"KV Cache NOPE verification failed: {nope_pcc_message}"
        assert rope_passing, f"KV Cache ROPE verification failed: {rope_pcc_message}"

    logger.info("✓ PreSDPA mesh test passed!")

    # Clean up trace and sub-device state before fixture teardown
    # This ensures profiler data is properly flushed before close_mesh_device
    ttnn.synchronize_device(submesh)
