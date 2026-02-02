# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN PreSDPA Test
Tests pre-SDPA fused operation with full pipeline:
- RMSNorm -> Matmul -> Gather -> RMSNorm2 -> Matmul2 (shuffled) -> Matmul3 (Qnope only) & RoPE (Qrope only) -> Interleaved Pre-SDPA Output
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


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True])
def test_pre_sdpa(device, epsilon, use_fp32):
    """Test TTNN pre-SDPA fused operation with full Qnope/Qrope pipeline"""

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

    # Device grid configuration
    device_grid_size = device.compute_with_storage_grid_size()

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

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
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
        device=device,
        memory_config=matmul_mem_config,
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
        device=device,
        memory_config=matmul2_mem_config,
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
        device=device,
        memory_config=rmsnorm2_gamma_mem_config,
        tile=tile,
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
        device=device,
        memory_config=matmul3_mem_config,
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
        device=device,
        memory_config=sdpa_input_output_mem_config,
        tile=tile,
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
        device=device,
        memory_config=qrope_cos_sin_mem_config,
        tile=tile,
    )

    ttnn_sin = ttnn.from_torch(
        sin_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qrope_cos_sin_mem_config,
        tile=tile,
    )

    # Trans_mat: [1, 1, 32, 32] - repeat for all qrope cores
    # Each core gets full [32, 32] transformation matrix (reused for all heads)
    trans_mat_replicated = torch_trans_mat.repeat(1, 1, qrope_num_cores, 1)  # [1, 1, 32, 32]
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    trans_shard_shape = (ttnn.TILE_SIZE, ttnn.TILE_SIZE)  # [32, 32] per core
    trans_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qrope_grid}),
        trans_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    ttnn_trans_mat = ttnn.from_torch(
        trans_mat_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
    )

    # ========================================================================
    # Run pre-SDPA operation
    # ========================================================================
    logger.info("Running pre-SDPA operation...")
    ttnn_sdpa_input_result = PreSDPA.op(
        ttnn_input,
        ttnn_gamma,
        ttnn_matmul_weights,
        ttnn_rmsnorm2_gamma,
        ttnn_matmul2_weights,
        ttnn_matmul3_weights,
        ttnn_sin,
        ttnn_cos,
        ttnn_trans_mat,
        ttnn_sdpa_input_output,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for verification
    sdpa_input_output_torch = ttnn.to_torch(ttnn_sdpa_input_result)

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    # Golden uses shuffled weights to produce same interleaved output
    _, _, torch_sdpa_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights_shuffled,  # Use shuffled weights
        torch_matmul3_weights,
        torch_sin,
        torch_cos,
        position_ids,
        epsilon=epsilon,
        num_qnope_heads=NUM_QNOPE_HEADS,
        num_qrope_heads=NUM_QROPE_HEADS,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
    )

    # ========================================================================
    # Verify final output (SDPA Input)
    # ========================================================================
    logger.info("Verifying SDPA Input interleaved results...")
    # Golden SDPA Input shape: [8, 8, 576] -> reshape to [8, 4608] to match device output
    # The 4×2 grid with ROW_MAJOR orientation gives indices 0-7 matching source rows 0-7
    # Each combined head is (qnope[512], qrope[64]) where qrope has been processed by RoPE
    torch_sdpa_input_expected_flat = torch_sdpa_expected.reshape(SDPA_INPUT_NUM_CORES, SDPA_INPUT_ELEMENTS_PER_CORE)

    sdpa_input_passing, sdpa_input_pcc_message = comp_pcc(torch_sdpa_input_expected_flat, sdpa_input_output_torch, 0.98)
    logger.info(f"SDPA Input PCC: {sdpa_input_pcc_message}")

    # Assert final output passes
    assert sdpa_input_passing, f"SDPA Input verification failed: {sdpa_input_pcc_message}"

    logger.info("✓ PreSDPA test passed!")
