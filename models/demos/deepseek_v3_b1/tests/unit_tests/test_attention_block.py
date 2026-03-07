# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Attention Block Test
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
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    BlitzDecodeWeights,
)
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.tests.unit_tests.test_post_sdpa import compute_forwarder_scratch_size
from models.demos.deepseek_v3_b1.tests.unit_tests.test_pre_sdpa import deinterleave_kv_cache


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
@pytest.mark.parametrize("bcast_cluster_axis", [0])
@pytest.mark.parametrize("bcast_secondary_cluster_axis", [1])
@pytest.mark.parametrize("reduce_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize(
    "position_id",
    [
        0,
        127,
        511,
        1023,
        2047,
        4096,  # (1 + partial,1,1,1): partial into dev0 (if SP = 4)
        pytest.param(6644, marks=pytest.mark.skip_post_commit),  # (2,2,1 + partial,1): partial into dev2 (if SP = 4)
        pytest.param(9916, marks=pytest.mark.skip_post_commit),  # (3,2 + partial,2,2): partial into dev1 (if SP = 4)
        pytest.param(11664, marks=pytest.mark.skip_post_commit),  # (3,3,3,2 + partial): partial into dev3 (if SP = 4)
    ],
)  # Must test 128 chunk aligned decode positions, add other tests when causal masks are in for SDPA
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
def test_attention_block(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    bcast_cluster_axis,
    bcast_secondary_cluster_axis,
    reduce_cluster_axis,
    num_iters,
    max_seq_len,
    position_id,
    noc_mode,
):
    """Test TTNN attention block fused operation with CCL broadcast, kv cache, mla, reduce, residual add"""
    num_devices = mesh_rows * mesh_cols
    skip_ccl = False
    # skip_ccl is not supported in this test

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh used by the test
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Configure a single worker sub-device covering the full compute grid
    device_grid_size = submesh.compute_with_storage_grid_size()

    attention_block_semaphores = AttentionBlock.create_semaphores(submesh)

    # ========================================================================
    # Configuration
    # ========================================================================
    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    per_device_max_seq_len = max_seq_len // mesh_rows
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

    M = 1
    K1 = QNOPE_OUT_DIM
    post_sdpa_intermediate = 8192
    K2 = 8192
    output_size = 7168

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

    # TODO: Reduce to 3 slots
    # SDPA output intermediate tensor declared here to overlap with remaining pre-SDPA CBs.
    # Matches flash_mla's cb_out_im sizing: 68 tiles of [8, 32] at bfloat16 = 43520 B per core.
    # Shard shape (32, 544) = 4 tile-rows x 17 tile-cols = 68 tiles per core.
    sdpa_out_interm_num_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_num_slots = 4
    sdpa_out_interm_shard_height = sdpa_out_interm_num_slots * 8  # 4 tile-rows of [8, 32]
    sdpa_out_interm_shard_width = 17 * 32  # 17 tile-cols of [8, 32]
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

    # DKV matmul weights (raw, unshuffled — BlitzDecodeWeights handles shard reordering)
    torch_dkv_matmul_weights = torch.randn(dkv_matmul_weights_shape, dtype=torch.bfloat16)

    # Placeholder tensors for get_tt_o_proj_and_gate_mm_weights (not consumed by pre-SDPA)
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

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)

    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    # kv_b2_proj: [K1, intermediate * num_tp] — full TP width, split along output dim per column
    torch_kv_b2_proj_weights = torch.randn((K1, post_sdpa_intermediate * num_tp), dtype=torch.bfloat16)
    # o_proj: [K2 * num_tp, output_size] — full TP height, split along input dim per column
    torch_o_proj_weights = torch.randn((K2 * num_tp, output_size), dtype=torch.bfloat16)

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
    matmul3_weights_overlapped, kv_b2_overlapped = bdw.get_tt_kv_b12_proj_weights(
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

    hidden_tile = ttnn.Tile([1, 32])  # Tilize tile shape for CreateQHeads output
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
    # torch_sdpa_output = torch.zeros(shape, dtype=torch.bfloat16)
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
        o_proj_overlapped,  # o_proj
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
    logger.info(f"Creating KV cache with per-device seq_len={per_device_max_seq_len}, total_seq_len={max_seq_len}...")
    kvpe_dim = KNOPE_DIM + KROPE_DIM
    cache_shape = (1, 1, max_seq_len, kvpe_dim)

    dcs = program_config.device_chunk_size
    num_sp = mesh_rows

    torch_kv_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    torch_kv_cache[:, :, :position_id, :] = torch.randn(1, 1, position_id, kvpe_dim, dtype=torch.bfloat16)
    torch_kv_cache_shuffled = deinterleave_kv_cache(torch_kv_cache, dcs, num_sp)

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
        torch_kv_cache_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None)),
    )
    kv_cache_bfp8_before_op = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Post-SDPA setup
    # Set up sub-device (not supported in slow dispatch mode)
    compute_grid_size = submesh.compute_with_storage_grid_size()
    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input/activation
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights
    sdpa_tile = ttnn.Tile([8, 32])  # 8x32 tiles for SDPA L and MS tensors (matches original SDPA op)

    # ========================================================================
    # Grid configuration — derived from production weight overlap specs
    # ========================================================================
    num_tp = mesh_cols

    kv_b12_cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    o_proj_cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC

    # Matmul1 grid: kv_b2 cores (5×8 + 12×2 = 64 cores)
    matmul1_grid = kv_b12_cfg.kv_b2_core_range_set
    num_matmul1_cores = matmul1_grid.num_cores()  # 64

    # Active Matmul2 cores: o_proj cores (12×8 + 8×2 = 112 cores)
    matmul2_grid = o_proj_cfg.o_proj_core_range_set
    num_matmul2_cores = matmul2_grid.num_cores()  # 112

    # SDPA configuration (matching original sdpa_reduce_to_all test)
    NUM_SDPA_WORKERS = 8
    SDPA_L_HEIGHT = 8  # 8 rows in L tensor (matches 8x32 tile)
    SDPA_L_WIDTH = 512 * NUM_SDPA_WORKERS  # 512 per worker = 4096 total
    SDPA_MS_WIDTH = 32 * NUM_SDPA_WORKERS  # 32 per worker = 256 total
    per_device_chunk_size = 1024

    # Per-core dimensions
    n1_per_core = post_sdpa_intermediate // num_matmul1_cores  # 8192 / 64 = 128
    n2_per_core = output_size // num_matmul2_cores  # 7168 / 112 = 64

    logger.info(f"Testing post_sdpa fused op with SDPA reduce-to-all phase:")
    logger.info(f"  SDPA: [{SDPA_L_HEIGHT}, {SDPA_L_WIDTH}] L tensor, [{SDPA_L_HEIGHT}, {SDPA_MS_WIDTH}] MS tensor")
    logger.info(f"  Matmul1: [{M}, {K1}] x [{K1}, {post_sdpa_intermediate}] on {num_matmul1_cores} cores")
    logger.info(f"  Matmul2: [{M}, {K2}] x [{K2}, {output_size}] on {num_matmul2_cores} active cores")
    logger.info(f"  TP All-Reduce: [{M}, {output_size}] across {num_devices} devices")

    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    sdpa_output_cores = FlashMLADecode.ProgramConfig.grid.output_cores(0, NUM_SDPA_WORKERS)
    sdpa_worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sdpa_output_cores]
    )

    # SDPA forwarder cores: (9,8), (10,8) — inside the kv_b2 matmul1 grid (cols 0-11, rows 8-9)
    # Must match the cores where sdpa_forwarder_scratch_mesh is allocated.
    sdpa_forwarder_cores = [ttnn.CoreCoord(9, 8), ttnn.CoreCoord(10, 8)]
    sdpa_forwarder_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(sdpa_forwarder_cores[0], sdpa_forwarder_cores[0]),
            ttnn.CoreRange(sdpa_forwarder_cores[1], sdpa_forwarder_cores[1]),
        ]
    )

    # SDPA input tensors per device: L [8, 4096], MS [8, 256]
    # MS tensor layout: for each worker core c (0..7), within columns [c*32, (c+1)*32]:
    #   column 0 = M (max), column 1 = S (sum), rest unused
    # This matches the original sdpa_reduce_to_all test/op format.
    MS_WIDTH_PER_CORE = 32  # ms_width per SDPA worker core
    device_sdpa_l_inputs = []
    device_sdpa_ms_inputs = []
    device_sdpa_m_inputs = []  # M (max) per device: [8, num_workers]
    device_sdpa_s_inputs = []  # S (sum) per device: [8, num_workers]
    for device_idx in range(num_devices):
        torch_sdpa_l = torch.randn((SDPA_L_HEIGHT, SDPA_L_WIDTH), dtype=torch.bfloat16)
        torch_sdpa_ms = torch.randn((SDPA_L_HEIGHT, SDPA_MS_WIDTH), dtype=torch.bfloat16)

        # Create structured M and S values at correct positions
        m_device = torch.zeros((SDPA_L_HEIGHT, NUM_SDPA_WORKERS), dtype=torch.bfloat16)
        s_device = torch.zeros((SDPA_L_HEIGHT, NUM_SDPA_WORKERS), dtype=torch.bfloat16)
        for core_idx in range(NUM_SDPA_WORKERS):
            m_device[:, core_idx] = torch.randn(SDPA_L_HEIGHT, dtype=torch.bfloat16) * 0.5 - 1.0
            s_device[:, core_idx] = torch.abs(torch.randn(SDPA_L_HEIGHT, dtype=torch.bfloat16)) + 0.1
            torch_sdpa_ms[:, core_idx * MS_WIDTH_PER_CORE + 0] = m_device[:, core_idx]
            torch_sdpa_ms[:, core_idx * MS_WIDTH_PER_CORE + 1] = s_device[:, core_idx]

        device_sdpa_l_inputs.append(torch_sdpa_l)
        device_sdpa_ms_inputs.append(torch_sdpa_ms)
        device_sdpa_m_inputs.append(m_device)
        device_sdpa_s_inputs.append(s_device)

    # ========================================================================
    # Create mesh mapper
    # ========================================================================
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)

    # ========================================================================
    # Create input tensor (height-sharded across matmul1 cores)
    # When SDPA is enabled, this tensor receives scatter output from SDPA workers
    # ========================================================================
    # input_shard_shape = (M, K1)  # [1, 512] per core
    # input_shard_spec = ttnn.ShardSpec(
    #    matmul1_grid,
    #    input_shard_shape,
    #    ttnn.ShardOrientation.ROW_MAJOR,
    # )
    # input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Initialize with zeros - SDPA scatter will populate this
    # torch_input_zeros = torch.zeros((num_matmul1_cores, K1), dtype=torch.bfloat16)
    # mesh_input_torch = torch.cat([torch_input_zeros] * num_devices, dim=0)
    # ttnn_input = ttnn.from_torch(
    #    mesh_input_torch,
    #    device=submesh,
    #    dtype=in0_dtype,
    #    layout=ttnn.TILE_LAYOUT,
    #    memory_config=input_mem_config,
    #    tile=a_tile,
    #    mesh_mapper=mesh_mapper,
    # )
    # logger.info(f"Created input tensor: shard {input_shard_shape} on {num_matmul1_cores} cores per device")

    # ========================================================================
    # Create CCL tensors and semaphores
    # ========================================================================
    output_shard_spec = ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    mesh_output_torch = torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0)
    ttnn_attention_block_output = ttnn.from_torch(
        mesh_output_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # ========================================================================
    # Create SDPA tensors
    # ========================================================================
    # SDPA L input tensor: [8, 4096] width-sharded across 8 SDPA workers
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS  # 512 per worker
    sdpa_l_shard_shape = (SDPA_L_HEIGHT, sdpa_l_per_worker)  # [8, 512]
    sdpa_l_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_l_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_l_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_l_shard_spec)

    mesh_sdpa_l_torch = torch.cat(device_sdpa_l_inputs, dim=0)
    ttnn_sdpa_input_l = ttnn.from_torch(
        mesh_sdpa_l_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created SDPA L input tensor: shard {sdpa_l_shard_shape} on {NUM_SDPA_WORKERS} workers")

    # SDPA MS input tensor: [8, 256] width-sharded across 8 SDPA workers
    sdpa_ms_per_worker = SDPA_MS_WIDTH // NUM_SDPA_WORKERS  # 32 per worker
    sdpa_ms_shard_shape = (SDPA_L_HEIGHT, sdpa_ms_per_worker)  # [8, 32]
    sdpa_ms_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_ms_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_ms_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_ms_shard_spec
    )

    mesh_sdpa_ms_torch = torch.cat(device_sdpa_ms_inputs, dim=0)
    ttnn_sdpa_input_ms = ttnn.from_torch(
        mesh_sdpa_ms_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_ms_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created SDPA MS input tensor: shard {sdpa_ms_shard_shape} on {NUM_SDPA_WORKERS} workers")

    # SDPA L output tensor (same shape as input)
    mesh_sdpa_l_out_torch = torch.zeros_like(mesh_sdpa_l_torch)
    ttnn_sdpa_output_l = ttnn.from_torch(
        mesh_sdpa_l_out_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )

    # SDPA R1/R2 receive buffers: must hold BOTH L and MS data per core.
    # Buffer layout per core: [L_chunk0][L_chunk1]...[L_chunkN-1][MS]
    # The original sdpa_reduce_to_all op uses shard shape [batch, l_width + ms_width] per core.
    # Using only L-sized buffers causes MS writes to go past the allocated memory!
    sdpa_recv_per_worker = sdpa_l_per_worker + sdpa_ms_per_worker  # 512 + 32 = 544
    sdpa_recv_shard_shape = (2 * SDPA_L_HEIGHT, sdpa_recv_per_worker)  # [8, 544]
    sdpa_recv_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_recv_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_recv_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_recv_shard_spec
    )
    # Full receive tensor: [8, (l_width + ms_width) * num_workers] = [8, 544*8] = [8, 4352]
    sdpa_recv_full_width = sdpa_recv_per_worker * NUM_SDPA_WORKERS
    mesh_sdpa_recv_torch = torch.cat(
        [torch.zeros((2 * SDPA_L_HEIGHT, sdpa_recv_full_width), dtype=torch.bfloat16)] * num_devices, dim=0
    )
    ttnn_sdpa_intermediate_recv = ttnn.from_torch(
        mesh_sdpa_recv_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_recv_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )

    # SDPA forwarder scratch buffer
    # Compute proper size using the same formula as sdpa_reduce_to_all op
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS  # 512 per worker
    sdpa_fwd_buffer_bytes = compute_forwarder_scratch_size(
        batch_size=SDPA_L_HEIGHT,
        l_width=sdpa_l_per_worker,
        num_cores=NUM_SDPA_WORKERS,
    )
    # Total elements (bfloat16 = 2 bytes per element)
    sdpa_fwd_total_elements = sdpa_fwd_buffer_bytes // 2
    # WIDTH_SHARDED across 2 forwarder cores, each gets half
    num_forwarders = 2
    sdpa_fwd_per_forwarder = sdpa_fwd_total_elements // num_forwarders
    sdpa_forwarder_shard_shape = (1, sdpa_fwd_per_forwarder)
    sdpa_forwarder_shard_spec = ttnn.ShardSpec(
        sdpa_forwarder_grid, sdpa_forwarder_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    sdpa_forwarder_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_forwarder_shard_spec
    )

    torch_forwarder_scratch = torch.zeros((1, sdpa_fwd_total_elements), dtype=torch.bfloat16)
    mesh_forwarder_scratch = torch.cat([torch_forwarder_scratch] * num_devices, dim=0)
    ttnn_sdpa_forwarder_scratch = ttnn.from_torch(
        mesh_forwarder_scratch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sdpa_forwarder_mem_config,
        mesh_mapper=mesh_mapper,
    )
    logger.info(
        f"Created SDPA forwarder scratch buffer: {sdpa_fwd_buffer_bytes} bytes total, {sdpa_fwd_per_forwarder} elements per forwarder"
    )
    # ========================================================================
    # Run pre-SDPA operation
    # ========================================================================
    logger.info(f"Running attention block operation with position_id={position_id}...")
    for i in range(num_iters):
        ttnn_output_result = AttentionBlock.op(
            input_tensor_mesh,
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
            # Post-SDPA parameters
            kv_b2_overlapped,
            o_proj_overlapped,
            #            ttnn_residual,
            ttnn_sdpa_input_l,
            ttnn_sdpa_input_ms,
            ttnn_sdpa_output_l,
            ttnn_sdpa_intermediate_recv,
            ttnn_sdpa_forwarder_scratch,
            program_config.device_chunk_size,  # sdpa_per_device_chunk_size
            ttnn_attention_block_output,
            # Shared semaphores, and some default values
            attention_block_semaphores,
            bcast_cluster_axis,
            bcast_secondary_cluster_axis,
            reduce_cluster_axis,
            0,  # sdpa_cluster_axis
            1.0,  # sdpa_scale_fp32
            1,  # num_links
            epsilon,
            use_fp32,
            skip_ccl,
            noc_mode,
        )
    ttnn.synchronize_device(submesh)

    kv_cache_output_torch = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # If True, validate the local FlashMLA output before SDPA AllReduce
    # Need to uncomment and use ttnn_output_tensor as the backing buffer for mla_out_o_cb
    validate_local_flash_mla = False

    # Read back the FlashMLA output (pre-post-SDPA) for validation
    if validate_local_flash_mla:
        sdpa_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_output_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    device_chunk_size = program_config.device_chunk_size
    num_sp = mesh_rows
    owning_sp_device = (position_id // device_chunk_size) % num_sp

    # Single golden call: full KV cache, global position_id.
    # Returns the new KV entry (for owning-device validation) and the global
    # SDPA output (post-SDPA reduces all SP local outputs to this).
    _, golden_new_kv, torch_output_expected = AttentionBlock.golden(
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
        torch_kv_b2_proj_weights,
        torch_o_proj_weights,
        epsilon=epsilon,
        num_qnope_heads=total_qnope_heads,
        num_qrope_heads=total_qrope_heads,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
    )

    logger.info(f"Golden computed (owning_sp_device={owning_sp_device}, device_chunk_size={device_chunk_size})")

    def get_local_seq_len(sp_idx):
        """Return how many KV positions SP device sp_idx holds for the current global position_id."""
        sp_block = device_chunk_size * num_sp
        num_full_blocks = position_id // sp_block
        remainder = position_id % sp_block
        dev_start = sp_idx * device_chunk_size
        dev_end = dev_start + device_chunk_size
        dev_contrib = max(0, min(remainder, dev_end) - dev_start)
        return num_full_blocks * device_chunk_size + dev_contrib

    if validate_local_flash_mla:
        # ========================================================================
        # Compute per-SP golden for pre-SDPA output (before post-SDPA reduce)
        # ========================================================================
        def build_local_kv_cache(sp_idx):
            """Extract sp_idx's local KV cache from torch_kv_cache_shuffled."""
            sp_block = device_chunk_size * num_sp
            num_full_blocks = position_id // sp_block
            remainder = position_id % sp_block
            dev_start = sp_idx * device_chunk_size
            dev_end = dev_start + device_chunk_size
            dev_contrib = max(0, min(remainder, dev_end) - dev_start)
            local_len = num_full_blocks * device_chunk_size + dev_contrib
            if local_len == 0:
                return None, 0
            shard_offset = sp_idx * per_device_max_seq_len
            local_kv = torch_kv_cache_shuffled[:, :, shard_offset : shard_offset + local_len, :]
            return local_kv, local_len

        pre_sdpa_golden_args = dict(
            input_tensor=torch_input,
            gamma_tensor=torch_gamma,
            matmul_weights_tensor=torch_matmul_weights,
            rmsnorm2_gamma_tensor=torch_rmsnorm2_gamma,
            matmul2_weights_tensor=torch_matmul2_weights_full_unshuffled,
            matmul3_weights_tensor=torch_matmul3_weights,
            sin_tensor=torch_sin,
            cos_tensor=torch_cos,
            dkv_matmul_weights_tensor=torch_dkv_matmul_weights,
            dkv_rmsnorm_gamma_tensor=torch_dkv_rmsnorm_gamma,
            scale=scale,
            epsilon=epsilon,
            num_qnope_heads=total_qnope_heads,
            num_qrope_heads=total_qrope_heads,
            qnope_head_dim=QNOPE_HEAD_DIM,
            qrope_head_dim=QROPE_HEAD_DIM,
            heads_per_row=HEADS_PER_ROW,
            nope_dim=KNOPE_DIM,
            rope_dim=KROPE_DIM,
        )

        golden_per_sp = {}
        for sp_idx in range(num_sp):
            local_kv, local_seq_len = build_local_kv_cache(sp_idx)
            is_owner = sp_idx == owning_sp_device
            if local_kv is None:
                if is_owner:
                    local_kv = torch.zeros(1, 1, 0, kvpe_dim, dtype=torch.bfloat16)
                else:
                    continue
            local_pos = torch.tensor([local_seq_len if is_owner else local_seq_len - 1])
            _, _, mla_output = PreSDPA.golden(
                **pre_sdpa_golden_args,
                local_position_ids=local_pos,
                global_position_ids=torch.tensor([position_id]),
                kv_cache_tensor=local_kv,
                kv_cache_update=is_owner,
            )
            golden_per_sp[sp_idx] = mla_output

        logger.info(
            f"Per-SP golden computed for {len(golden_per_sp)} SP devices "
            f"(owning_sp_device={owning_sp_device}, device_chunk_size={device_chunk_size})"
        )

    # ========================================================================
    # Validate KV cache outputs (per SP device)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        sp_group = device_idx // mesh_cols
        local_seq_len = get_local_seq_len(sp_group)

        if local_seq_len == 0 and sp_group != owning_sp_device:
            logger.info(f"Device {device_idx} (SP={sp_group}) no data yet, skipped")
            continue

        # ---- KV Cache: old positions must be unchanged ----
        assert torch.equal(
            kv_cache_bfp8_before_op[device_idx, ..., :local_seq_len, :],
            kv_cache_output_torch[device_idx, ..., :local_seq_len, :],
        ), f"Device {device_idx} (SP={sp_group}) KV Cache before and after op mismatch"
        logger.info(f"Device {device_idx} (SP={sp_group}) old cache validation passed")

        # ---- KV Cache new-position check (owning device only) ----
        if sp_group == owning_sp_device:
            compare_kv_cache = kv_cache_output_torch[device_idx, ..., local_seq_len, :]
            expected_nope = golden_new_kv[..., :KNOPE_DIM]
            expected_rope = golden_new_kv[..., KNOPE_DIM:]
            compare_nope = compare_kv_cache[..., :KNOPE_DIM]
            compare_rope = compare_kv_cache[..., KNOPE_DIM:]

            nope_passing, nope_pcc = comp_pcc(compare_nope, expected_nope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC: {nope_pcc}")
            assert nope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC check failed: {nope_pcc}"

            rope_passing, rope_pcc = comp_pcc(compare_rope, expected_rope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC: {rope_pcc}")
            assert rope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC check failed: {rope_pcc}"

    # ========================================================================
    # Validate pre-SDPA output (per-SP golden: local KV cache per device)
    # This is the FlashMLA output before post-SDPA reduce across SP devices.
    # ========================================================================
    if validate_local_flash_mla:
        slice_size = sdpa_input_output_shape[0]  # 64 heads per device
        for device_idx in range(mesh_rows * mesh_cols):
            tp_group = device_idx % mesh_cols
            sp_group = device_idx // mesh_cols

            if sp_group not in golden_per_sp:
                logger.info(f"Device {device_idx} (SP={sp_group}) no work, skipped")
                continue

            golden_mla_output = golden_per_sp[sp_group]

            start = device_idx * slice_size
            end = start + slice_size
            received = sdpa_output_torch[start:end, :]

            tp_start = tp_group * slice_size
            tp_end = tp_start + slice_size
            expected = golden_mla_output[tp_start:tp_end, :]
            print(expected[:8, :32:8])

            if received.shape != expected.shape:
                logger.error(
                    f"Device {device_idx} (TP={tp_group}, SP={sp_group}) output shape mismatch: "
                    f"got {received.shape}, expected {expected.shape}"
                )
                continue

            passing, pcc = comp_pcc(expected, received, 0.84)
            logger.info(f"Device {device_idx} (TP={tp_group}, SP={sp_group}) PreSDPA Output PCC: {pcc}")
            assert passing, f"Device {device_idx} (TP={tp_group}, SP={sp_group}) PreSDPA Output PCC check failed: {pcc}"

    # ========================================================================
    # Validate attention block output (full pipeline: SDPA -> kv_b2 -> o_proj -> all-reduce + residual)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        received = output_torch[device_idx : device_idx + 1, :]
        passing, pcc = comp_pcc(torch_output_expected, received, 0.88)
        logger.info(f"Device {device_idx} Attention Block Output PCC: {pcc}")
        assert passing, f"Device {device_idx} Attention Block Output PCC check failed: {pcc}"

    logger.info("✓ Attention Block mesh test passed!")

    # Clean up trace and sub-device state before fixture teardown
    # This ensures profiler data is properly flushed before close_mesh_device
    ttnn.synchronize_device(submesh)
