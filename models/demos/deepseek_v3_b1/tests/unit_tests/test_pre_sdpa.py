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
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode


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
@pytest.mark.parametrize("position_id", [0, 1, 127, 242, 255, 564, 1023])
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
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
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
    noc_mode,
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

    semaphores = PreSDPA.create_semaphores(submesh, skip_ccl)

    # ========================================================================
    # Configuration
    # ========================================================================
    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    max_seq_len = 32 * 1024
    assert position_id < max_seq_len, f"Position ID {position_id} must be less than max sequence length {max_seq_len}"

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

    scale = (QNOPE_HEAD_DIM + QROPE_HEAD_DIM) ** -0.5

    # Qnope/Qrope grid configuration (must match head configuration)
    QNOPE_GRID_COLS = 8  # 8 Qnope cores per row (1 head each)
    QROPE_GRID_COLS = 4  # 4 Qrope cores per row (2 heads each)
    TOTAL_COLS = QNOPE_GRID_COLS + QROPE_GRID_COLS  # 12 columns required

    # Skip test on P100 devices (which have 11 columns instead of 12)
    if device_grid_size.x < TOTAL_COLS:
        pytest.skip(
            f"Device grid {device_grid_size.x}x{device_grid_size.y} too small for {TOTAL_COLS} columns required (P100 has 11 columns)"
        )

    matmul2_grid_x = 12
    matmul2_grid_y = 8

    # SDPA KV Cache tensor declared here to overlap with pre-SDPA, pre-KV cache branch CBs.
    # Matches flash_mla's double-buffered KV CB sizing: shard = (2 * k_chunk_size, kvpe_dim)
    # = (256, 576) per core, giving 156,672 bytes/core — same as flash_mla's cb_k_in.
    # Total height = shard_height * num_cores must be divisible by tile height (32).
    kvpe_dim = KNOPE_DIM + KROPE_DIM  # 576
    kv_cache_num_cores_x = device_grid_size.x
    kv_cache_num_cores_y = device_grid_size.y
    kv_cache_num_cores = kv_cache_num_cores_x * kv_cache_num_cores_y
    kv_cache_shard_height = 256  # 2 * k_chunk_size (128), matching flash_mla double-buffer
    kv_cache_total_height = kv_cache_shard_height * kv_cache_num_cores

    kv_cache_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(kv_cache_num_cores_x - 1, kv_cache_num_cores_y - 1),
                )
            }
        ),
        (kv_cache_shard_height, kvpe_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.randn((kv_cache_total_height, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            kv_cache_shard_spec,
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # SDPA output intermediate tensor declared here to overlap with remaining pre-SDPA CBs.
    # Matches flash_mla's cb_out_im sizing: 85 tiles of [8, 32] at bfloat16 = 43520 B per core.
    # Shard shape (40, 544) = 5 tile-rows x 17 tile-cols = 85 tiles per core.
    sdpa_out_interm_num_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_height = 40  # 5 tile-rows of [8, 32]
    sdpa_out_interm_shard_width = 544  # 17 tile-cols of [8, 32]
    sdpa_out_interm_total_height = sdpa_out_interm_shard_height * sdpa_out_interm_num_cores

    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1),
                )
            }
        ),
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_total_height, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        tile=ttnn.Tile([8, 32]),
    )

    # Matmul2 output width (interleaved Qnope/Qrope)
    matmul2_width = NUM_QNOPE_HEADS * QNOPE_HEAD_DIM + NUM_QROPE_HEADS * QROPE_HEAD_DIM  # 8192 + 4096 = 12288
    matmul2_weights_shape = (1536, matmul2_width)
    num_tp = mesh_cols  # TP parallelism across mesh columns (2 for 4x2, 1 for 1x1)
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

    # Matmul2 weights - full tensor with layout [all_qnope | all_qrope] for num_tp * 64 heads.
    # Golden receives this directly; per-TP slices are extracted for shuffling + mesh distribution.
    total_qnope_heads = num_tp * NUM_QNOPE_HEADS
    total_qrope_heads = num_tp * NUM_QROPE_HEADS
    total_qnope_dim = total_qnope_heads * QNOPE_HEAD_DIM
    total_qrope_dim = total_qrope_heads * QROPE_HEAD_DIM
    torch_matmul2_weights_full_unshuffled = torch.randn(
        (matmul2_weights_shape[0], total_qnope_dim + total_qrope_dim), dtype=torch.bfloat16
    )

    # Matmul3 weights - [num_tp * num_qnope_heads, qnope_head_dim, qnope_out_dim] for golden
    # Each TP slice of 64 heads is height-sharded on 64 cores per device.
    torch_matmul3_weights = torch.randn((num_tp * NUM_QNOPE_HEADS, QNOPE_HEAD_DIM, QNOPE_OUT_DIM), dtype=torch.bfloat16)

    # kv_b2_proj weights (placeholder — not consumed by pre-SDPA but required by the fused buffer)
    torch_kv_b2_proj_weights = torch.zeros(
        (QNOPE_OUT_DIM, num_tp * NUM_QNOPE_HEADS * QNOPE_HEAD_DIM), dtype=torch.bfloat16
    )

    # DKV matmul weights (raw, unshuffled — BlitzDecodeWeights handles shard reordering)
    torch_dkv_matmul_weights = torch.randn(dkv_matmul_weights_shape, dtype=torch.bfloat16)

    # Placeholder tensors for get_tt_o_proj_and_gate_mm_weights (not consumed by pre-SDPA)
    torch_o_proj_weights = torch.zeros((num_tp * 8192, 7168), dtype=torch.bfloat16)
    torch_gate_mm_weights = torch.zeros((7168, 256), dtype=torch.bfloat16)
    torch_ffn_norm = torch.zeros((1, 7168), dtype=torch.bfloat16)

    # ========================================================================
    # Create RoPE tensors (sin, cos, trans_mat)
    # ========================================================================
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

    # Fused matmul1 (q_a_proj packed), matmul2 (q_b_proj shuffled), and DKV matmul (kv_a_proj)
    # weights as overlapped tensors sharing a single L1 buffer via BlitzDecodeWeights.
    bdw = BlitzDecodeWeights(submesh)
    (
        matmul_weights_overlapped,
        matmul2_weights_overlapped,
        dkv_matmul_weights_overlapped,
    ) = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(
        torch_matmul_weights,
        torch_matmul2_weights_full_unshuffled,
        torch_dkv_matmul_weights,
    )

    # Matmul3 / kv_b1_proj weights — fused with kv_b2_proj via BlitzDecodeWeights
    torch_matmul3_weights_flat = torch_matmul3_weights.reshape(num_tp * NUM_QNOPE_HEADS * QNOPE_HEAD_DIM, QNOPE_OUT_DIM)
    matmul3_weights_overlapped, _ = bdw.get_tt_kv_b12_proj_weights(
        torch_matmul3_weights_flat,
        torch_kv_b2_proj_weights,
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

    s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
    sdpa_input_output_grid_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores]
    )

    sdpa_tile = ttnn.Tile([8, 32])  # Tilize tile shape for CreateQHeads output
    sdpa_input_output_shape = (SDPA_INPUT_NUM_CORES * HEADS_PER_ROW, QNOPE_OUT_DIM)  # [64, 512] total
    sdpa_input_output_shard_shape = (HEADS_PER_ROW, QNOPE_OUT_DIM)  # [8, 512] per core
    sdpa_input_output_shard_spec = ttnn.ShardSpec(
        sdpa_input_output_grid_crs,
        sdpa_input_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_input_output_shard_spec
    )
    torch_sdpa_output = torch.zeros(sdpa_input_output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_sdpa_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=sdpa_mem_config,
        tile=sdpa_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # RoPE tensors
    qrope_grid = ttnn.CoreRange(
        ttnn.CoreCoord(QNOPE_GRID_COLS, 0), ttnn.CoreCoord(QNOPE_GRID_COLS + QROPE_GRID_COLS - 1, matmul2_grid_y - 1)
    )

    # QRoPE cos/sin: DRAM INTERLEAVED (all qrope cores read full head_dim)
    # Shape: [1, 1, max_seq_len, qrope_head_dim]
    qrope_cos_full = torch_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, 64]
    qrope_sin_full = torch_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, 64]

    qrope_dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    ttnn_qrope_cos = ttnn.from_torch(
        qrope_cos_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    ttnn_qrope_sin = ttnn.from_torch(
        qrope_sin_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem_config,
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

    # KV Cache Branch RMSNorm gamma
    torch_dkv_rmsnorm_gamma = torch.randn((1, KNOPE_DIM), dtype=torch.bfloat16)

    # Fused o_proj, gate_mm, and RMSNorm gammas — we only need the 3 gamma overlapped views.
    (
        _,  # o_proj
        _,  # gate_mm
        gamma_overlapped,
        rmsnorm2_gamma_overlapped,
        dkv_rmsnorm_gamma_overlapped,
        _,  # ffn_norm
    ) = bdw.get_tt_o_proj_and_gate_mm_weights(
        torch_o_proj_weights,
        torch_gate_mm_weights,
        torch_gamma,
        torch_rmsnorm2_gamma,
        torch_dkv_rmsnorm_gamma,
        torch_ffn_norm,
    )

    # KRoPE cos/sin: DRAM INTERLEAVED (each krope core reads its width slice)
    krope_num_cores = kv_cache_branch_rope_crs.num_cores()
    krope_cos_full = torch_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, 64]
    krope_sin_full = torch_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, 64]

    krope_dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    ttnn_krope_cos = ttnn.from_torch(
        krope_cos_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=krope_dram_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_krope_sin = ttnn.from_torch(
        krope_sin_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=krope_dram_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    position_replicated = torch.full((device_grid_size.x * device_grid_size.y, 1), position_id, dtype=torch.int32)
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )
    pos_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_position_ids = ttnn.from_torch(
        position_replicated,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=pos_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # KV Cache tensor in DRAM sharded
    # Create KV cache (non-paged) based on max seq len
    program_config = FlashMLADecode.ProgramConfig(
        k_chunk_size=128,
        exp_approx_mode=False,  # Use exact exp for higher precision
    )
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    kvpe_dim = KNOPE_DIM + KROPE_DIM
    cache_shape = (1, 1, max_seq_len, kvpe_dim)
    # from 0 to position id, the kv cache is valid, position_id data is filled by test
    torch_kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    for i in range(position_id):
        torch_kv_cache[:, :, i, :] = torch.randn(1, 1, 1, kvpe_dim, dtype=torch.bfloat16)

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
    kv_cache_bfp8_before_op = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # ========================================================================
    # Run pre-SDPA operation
    # ========================================================================
    logger.info("Running pre-SDPA operation...")

    for i in range(num_iters):
        ttnn_output_result = PreSDPA.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            gamma_overlapped,
            matmul_weights_overlapped,
            rmsnorm2_gamma_overlapped,
            matmul2_weights_overlapped,
            matmul3_weights_overlapped,
            ttnn_qrope_sin,
            ttnn_qrope_cos,
            ttnn_trans_mat,
            ttnn_krope_cos,
            ttnn_krope_sin,
            dkv_matmul_weights_overlapped,
            dkv_rmsnorm_gamma_overlapped,
            ttnn_kv_cache,
            position_id,
            ttnn_position_ids,
            scale,
            ttnn_output,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            sender_coord,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            secondary_cluster_axis=secondary_cluster_axis,
            epsilon=epsilon,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=skip_ccl,
            noc_mode=noc_mode,
        )
    ttnn.synchronize_device(submesh)

    kv_cache_output_torch = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_output_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    # Golden uses unshuffled weights (sequential output: all QNOPE, then all QROPE).
    # Full tensor with num_tp * 64 heads; output is split per-TP for comparison.
    torch_q_expected, torch_kv_cache_expected, torch_output_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights_full_unshuffled,
        torch_matmul3_weights,
        torch_sin,
        torch_cos,
        position_ids,
        torch_dkv_matmul_weights,
        torch_dkv_rmsnorm_gamma,
        torch_kv_cache,
        scale,
        epsilon=epsilon,
        num_qnope_heads=total_qnope_heads,
        num_qrope_heads=total_qrope_heads,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
    )

    slice_size = sdpa_input_output_shape[0]
    expected_width = KNOPE_DIM  # 512

    # KV Cache is same across devices in 4x2 submesh
    expected_nope = torch_kv_cache_expected[..., :KNOPE_DIM]
    expected_rope = torch_kv_cache_expected[..., KNOPE_DIM:]

    sdpa_pcc_by_tp = {}
    kv_nope_pcc_first = None
    kv_rope_pcc_first = None

    for device_idx in range(mesh_rows * mesh_cols):
        tp_group = device_idx % mesh_cols  # TP group determined by mesh column

        # ---- KV Cache (fully replicated, no TP) ----
        compare_kv_cache = kv_cache_output_torch[device_idx, ..., position_id, :]

        # check that kv cache for 0 to pos_id -  1 is identical to kv cache before op
        for i in range(position_id):
            assert torch.allclose(
                kv_cache_bfp8_before_op[device_idx, ..., i, :], kv_cache_output_torch[device_idx, ..., i, :], atol=1e-6
            ), "KV Cache before and after op mismatch"
        logger.info(f"Device {device_idx} old cache validation passed")

        # Check that the new kv cache for pos_id is correct compared to golden
        compare_nope = compare_kv_cache[..., :KNOPE_DIM]
        compare_rope = compare_kv_cache[..., KNOPE_DIM:]

        nope_max_diff = torch.max(torch.abs(expected_nope - compare_nope)).item()
        nope_mean_diff = torch.mean(torch.abs(expected_nope - compare_nope)).item()
        logger.info(f"Device {device_idx} KV Cache NOPE: Max diff={nope_max_diff}, Mean diff={nope_mean_diff}")
        nope_passing, nope_pcc = comp_pcc(compare_nope, expected_nope, 0.98)
        logger.info(f"Device {device_idx} KV Cache NOPE PCC: {nope_pcc}")
        assert nope_passing, f"Device {device_idx} KV Cache NOPE PCC check failed: {nope_pcc}"

        if kv_nope_pcc_first is not None:
            assert nope_pcc == kv_nope_pcc_first, (
                f"Device {device_idx} KV Cache NOPE PCC mismatch across replicated dim: "
                f"got {nope_pcc}, expected {kv_nope_pcc_first}"
            )
        else:
            kv_nope_pcc_first = nope_pcc
        rope_max_diff = torch.max(torch.abs(expected_rope - compare_rope)).item()
        rope_mean_diff = torch.mean(torch.abs(expected_rope - compare_rope)).item()
        logger.info(f"Device {device_idx} KV Cache ROPE: Max diff={rope_max_diff}, Mean diff={rope_mean_diff}")
        rope_passing, rope_pcc = comp_pcc(compare_rope, expected_rope, 0.98)
        logger.info(f"Device {device_idx} KV Cache ROPE PCC: {rope_pcc}")
        assert rope_passing, f"Device {device_idx} KV Cache ROPE PCC check failed: {rope_pcc}"

        if kv_rope_pcc_first is not None:
            assert rope_pcc == kv_rope_pcc_first, (
                f"Device {device_idx} KV Cache ROPE PCC mismatch across replicated dim: "
                f"got {rope_pcc}, expected {kv_rope_pcc_first}"
            )
        else:
            kv_rope_pcc_first = rope_pcc

        # ---- PreSDPA Output (TP-sharded: replicated across rows, different across columns) ----
        start = device_idx * slice_size
        end = start + slice_size
        received = output_torch[start:end, :expected_width]

        # Golden SDPA output is [num_tp * 64, 576]; slice per TP group.
        # Each row is one head: [qnope[512], qrope[64]], 8 cores x 8 heads = 64 rows per TP.
        tp_start = tp_group * slice_size
        tp_end = tp_start + slice_size
        torch_output_expected_flat = torch_output_expected[tp_start:tp_end, :]

        if received.shape != torch_output_expected_flat.shape:
            logger.error(
                f"PreSDPA Output shape mismatch at device {device_idx} (TP={tp_group}): "
                f"got {received.shape}, expected {torch_output_expected_flat.shape}"
            )
            continue

        max_diff = torch.max(torch.abs(received - torch_output_expected_flat)).item()
        mean_diff = torch.mean(torch.abs(received - torch_output_expected_flat)).item()
        logger.info(f"Device {device_idx} (TP={tp_group}) PreSDPA Output: Max diff={max_diff}, Mean diff={mean_diff}")

        # Lower PCC threshold due to random weights
        passing, sdpa_pcc = comp_pcc(torch_output_expected_flat, received, 0.90)
        logger.info(f"Device {device_idx} (TP={tp_group}) PreSDPA Output PCC: {sdpa_pcc}")
        assert passing, f"Device {device_idx} (TP={tp_group}) PreSDPA Output PCC check failed: {sdpa_pcc}"

        if tp_group in sdpa_pcc_by_tp:
            assert sdpa_pcc == sdpa_pcc_by_tp[tp_group], (
                f"Device {device_idx} (TP={tp_group}) PreSDPA Output PCC mismatch across replicated dim: "
                f"got {sdpa_pcc}, expected {sdpa_pcc_by_tp[tp_group]}"
            )
        else:
            sdpa_pcc_by_tp[tp_group] = sdpa_pcc

    logger.info("✓ PreSDPA mesh test passed!")

    # Clean up trace and sub-device state before fixture teardown
    # This ensures profiler data is properly flushed before close_mesh_device
    ttnn.synchronize_device(submesh)
