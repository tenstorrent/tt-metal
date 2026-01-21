# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN PreSDPA Test
Tests pre-SDPA fused operation with full pipeline:
- RMSNorm -> Matmul -> Gather -> RMSNorm2 -> Matmul2 (shuffled) -> Matmul3 (Qnope only)
- Qnope output: [64, 1, 512] after matmul3
- Qrope output: [64, 1, 64] (matmul2 output, ready for RoPE)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
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
    matmul2_grid_x = min(TOTAL_COLS, device_grid_size.x)  # Must be exactly 12 for correct sharding
    matmul2_grid_y = 8

    # Matmul2 output width (interleaved Qnope/Qrope)
    matmul2_width = NUM_QNOPE_HEADS * QNOPE_HEAD_DIM + NUM_QROPE_HEADS * QROPE_HEAD_DIM  # 8192 + 4096 = 12288
    matmul2_weights_shape = (1536, matmul2_width)
    qnope_num_cores = QNOPE_GRID_COLS * matmul2_grid_y  # 64 cores
    qrope_num_cores = QROPE_GRID_COLS * matmul2_grid_y  # 32 cores

    # Matmul3 weights shape: [128, 512] per Qnope core
    matmul3_weights_shape = (QNOPE_HEAD_DIM, QNOPE_OUT_DIM)  # Per-core shape

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

    # Qnope output tensor - height sharded on Qnope grid, [1, 512] per core
    qnope_output_shape = (qnope_num_cores, QNOPE_OUT_DIM)  # [64, 512] total
    qnope_output_shard_shape = (1, QNOPE_OUT_DIM)  # [1, 512] per core
    qnope_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qnope_grid}),
        qnope_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    qnope_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, qnope_output_shard_spec
    )
    torch_qnope_output = torch.zeros(qnope_output_shape, dtype=torch.bfloat16)
    ttnn_qnope_output = ttnn.from_torch(
        torch_qnope_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qnope_output_mem_config,
        tile=tile,
    )

    # Qrope output tensor - height sharded on Qrope grid, [1, 128] per core (2 heads × 64)
    qrope_grid_start_x = QNOPE_GRID_COLS  # Column 8
    qrope_grid_end_x = QNOPE_GRID_COLS + QROPE_GRID_COLS - 1  # Column 11
    qrope_grid = ttnn.CoreRange(
        ttnn.CoreCoord(qrope_grid_start_x, 0), ttnn.CoreCoord(qrope_grid_end_x, matmul2_grid_y - 1)
    )
    qrope_elements_per_core = 2 * QROPE_HEAD_DIM  # 2 heads × 64 = 128 elements per core
    qrope_output_shape = (qrope_num_cores, qrope_elements_per_core)  # [32, 128] total
    qrope_output_shard_shape = (1, qrope_elements_per_core)  # [1, 128] per core
    qrope_output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qrope_grid}),
        qrope_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    qrope_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, qrope_output_shard_spec
    )
    torch_qrope_output = torch.zeros(qrope_output_shape, dtype=torch.bfloat16)
    ttnn_qrope_output = ttnn.from_torch(
        torch_qrope_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qrope_output_mem_config,
        tile=tile,
    )

    logger.info(f"Created all tensors for pre-SDPA operation")
    logger.info(f"Qnope output shape: {qnope_output_shape}, shard: {qnope_output_shard_shape}")
    logger.info(f"Qrope output shape: {qrope_output_shape}, shard: {qrope_output_shard_shape}")

    # ========================================================================
    # Run pre-SDPA operation
    # ========================================================================
    logger.info("Running pre-SDPA operation...")
    ttnn_qnope_result, ttnn_qrope_result = PreSDPA.op(
        ttnn_input,
        ttnn_gamma,
        ttnn_matmul_weights,
        ttnn_rmsnorm2_gamma,
        ttnn_matmul2_weights,
        ttnn_matmul3_weights,
        ttnn_qnope_output,
        ttnn_qrope_output,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for verification
    qnope_output_torch = ttnn.to_torch(ttnn_qnope_result)
    qrope_output_torch = ttnn.to_torch(ttnn_qrope_result)

    logger.info(f"Device Qnope output shape: {qnope_output_torch.shape}")
    logger.info(f"Device Qrope output shape: {qrope_output_torch.shape}")

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    # Golden uses shuffled weights to produce same interleaved output
    torch_qnope_expected, torch_qrope_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights_shuffled,  # Use shuffled weights
        torch_matmul3_weights,
        epsilon=epsilon,
        num_qnope_heads=NUM_QNOPE_HEADS,
        num_qrope_heads=NUM_QROPE_HEADS,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
    )

    logger.info(f"Golden Qnope output shape: {torch_qnope_expected.shape}")
    logger.info(f"Golden Qrope output shape: {torch_qrope_expected.shape}")

    # Reshape golden outputs to match device output shapes
    # Golden: [num_heads, 1, head_dim] -> Device: [num_heads, head_dim] (no middle dim)
    torch_qnope_expected_flat = torch_qnope_expected.squeeze(1)  # [64, 512]
    torch_qrope_expected_flat = torch_qrope_expected.squeeze(1)  # [64, 64]

    # Qrope needs to be regrouped: [64, 64] -> [32, 128] (2 heads per core)
    torch_qrope_expected_grouped = torch_qrope_expected_flat.reshape(qrope_num_cores, qrope_elements_per_core)

    # ========================================================================
    # Verify results
    # ========================================================================
    logger.info("Verifying Qnope results...")
    qnope_passing, qnope_pcc_message = comp_pcc(torch_qnope_expected_flat, qnope_output_torch, 0.98)
    logger.info(f"Qnope PCC: {qnope_pcc_message}")

    logger.info("Verifying Qrope results...")
    qrope_max_diff = torch.max(torch.abs(qrope_output_torch - torch_qrope_expected_grouped)).item()
    qrope_mean_diff = torch.mean(torch.abs(qrope_output_torch - torch_qrope_expected_grouped)).item()
    logger.info(f"Qrope max absolute difference: {qrope_max_diff}")
    logger.info(f"Qrope mean absolute difference: {qrope_mean_diff}")

    qrope_passing, qrope_pcc_message = comp_pcc(torch_qrope_expected_grouped, qrope_output_torch, 0.98)
    logger.info(f"Qrope PCC: {qrope_pcc_message}")

    # Assert both pass
    assert qnope_passing, f"Qnope verification failed: {qnope_pcc_message}"
    assert qrope_passing, f"Qrope verification failed: {qrope_pcc_message}"

    logger.info("✓ PreSDPA test passed!")
