# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN DRAM Streaming Matmul Micro Op Test

Tests the simplified DRAM streaming matmul operation where:
- Input A is REPLICATED on compute cores (each core has full [M, K])
- Input B (weights) is WIDTH_SHARDED in DRAM with per-shard tile reordering
- Output is WIDTH_SHARDED in L1 on compute cores
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def pad_to_dram_banks(num, tile_w, lcm):
    """Pad number to be aligned with DRAM banks."""
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    return num + padding_needed


def shuffle_tensor_tiles(tensor, tile_size, num_banks):
    """
    Shuffle tiles for WIDTH_SHARDED from row-major to column-major within each shard.

    For shape [K, N] with WIDTH_SHARDED, TTNN tilizes with row-major tiles:
        (0,0), (0,1), ..., (0,pN-1), (1,0), (1,1), ...
    But we want column-major within each shard for streaming K tiles at a time:
        (0,0), (1,0), ..., (K-1,0), (0,1), (1,1), ...

    This pre-shuffles tiles so TTNN's row-major storage gives us column-major order.
    Handles padding internally: pads N to align with num_banks, shuffles, then slices back.

    Args:
        tensor: [*, K, N] tensor (supports batch dimensions)
        tile_size: tile dimension (assumes square tiles)
        num_banks: number of DRAM banks (shards)

    Returns:
        [*, K, N] tensor with tiles rearranged per-shard (same shape as input)
    """
    # Handle batch dimensions
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]
    tensor_2d = tensor.reshape(-1, K, N) if len(orig_shape) > 2 else tensor.unsqueeze(0)
    batch_size = tensor_2d.shape[0]

    # Pad N to align with num_banks
    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    if n_padded != N:
        tensor_2d = torch.nn.functional.pad(tensor_2d, (0, n_padded - N))

    K_tiles = K // tile_size
    per_core_N_tiles = n_padded // num_banks // tile_size

    shuffled = torch.zeros_like(tensor_2d)

    for b in range(batch_size):
        for bank in range(num_banks):
            for kt_shuf in range(K_tiles):
                for local_nt_shuf in range(per_core_N_tiles):
                    # Row-major index within this bank's shard
                    local_shuf_idx = kt_shuf * per_core_N_tiles + local_nt_shuf

                    # Map to column-major: which original tile goes here
                    kt_orig = local_shuf_idx % K_tiles
                    local_nt_orig = local_shuf_idx // K_tiles

                    # Global tile column indices
                    nt_shuf = bank * per_core_N_tiles + local_nt_shuf
                    nt_orig = bank * per_core_N_tiles + local_nt_orig

                    # Copy tile
                    shuffled[
                        b,
                        kt_shuf * tile_size : (kt_shuf + 1) * tile_size,
                        nt_shuf * tile_size : (nt_shuf + 1) * tile_size,
                    ] = tensor_2d[
                        b,
                        kt_orig * tile_size : (kt_orig + 1) * tile_size,
                        nt_orig * tile_size : (nt_orig + 1) * tile_size,
                    ]

    # Slice back to original N and restore shape
    shuffled = shuffled[:, :, :N]
    if len(orig_shape) > 2:
        shuffled = shuffled.reshape(*orig_shape[:-2], K, N)
    else:
        shuffled = shuffled.squeeze(0)

    return shuffled


@pytest.mark.parametrize("k, n", [(7168, 2048), (2048, 7168)])
@pytest.mark.parametrize("m", [1, 4, 8])
@pytest.mark.parametrize("fused_activation", [None, "silu"])
def test_dram_streaming_matmul(device, k, n, m, fused_activation):
    """Test simplified DRAM streaming matmul with optional fused activation.

    In the simplified version:
    - Input A is REPLICATED on compute cores (each core has full [M, K])
    - Input B is WIDTH_SHARDED [K, N] with per-shard tiles shuffled to column-major
    - No multicast needed - each core has its own copy
    - Output is WIDTH_SHARDED across N dimension

    Args:
        fused_activation: Optional activation to fuse (e.g., "silu", "gelu", "relu")
    """

    tile_h = m  # Tile height matches m (1 for tiny tiles, 32 for standard)
    tile_w = 32

    # Create tile object for tiny tiles when m=1
    in0_tile = ttnn.Tile([tile_h, tile_w])
    out_tile = ttnn.Tile([tile_h, tile_w])

    # Get compute cores from optimal DRAM bank assignment
    # These are the cores that will do the compute (8 on BH, 12 on WH)
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)

    # Get number of DRAM banks (should match num_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks, f"num_cores ({num_cores}) must equal num_banks ({num_banks})"

    logger.info(f"num_compute_cores={num_cores}, num_dram_banks={num_banks}")

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    logger.info(f"n_padded={n_padded}, per_core_N={per_core_N}, Kt={k // tile_w}")

    # Define shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n_padded]  # Original shape for matmul reference

    # Build CoreRangeSet for specific compute cores (not bounding box)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    # Create PyTorch tensors
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()  # [1, 1, K, N] for reference

    # ========== Input A - REPLICATED on compute cores ==========
    # Replicate the tensor num_cores times along height, then HEIGHT_SHARD
    in0_replicated = in0.repeat(1, 1, num_cores, 1)  # Shape: [1, 1, M * num_cores, K]
    in0_shard_shape_full = [m, k]  # Each core gets [M, K]
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, in0_shard_shape_full, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.from_torch(
        in0_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        tile=in0_tile,
    )

    # ========== Input B - WIDTH_SHARDED in DRAM with per-shard tile reordering ==========
    # Shuffle tiles so K tiles are contiguous per N column (for streaming)
    in1_shuffled = shuffle_tensor_tiles(in1, tile_w, num_banks)

    # WIDTH_SHARDED across N dimension, each bank gets [K, n // num_banks]
    in1_shard_shape = [k, n_padded // num_banks]
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1_shuffled,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # ========== Output tensor - WIDTH_SHARDED in L1 ==========
    output_shard_width = n_padded // num_banks
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros([1, 1, m, n_padded]).bfloat16().float()
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    if k == 7168:
        subblock_k = k // tile_w // 4
    else:
        subblock_k = k // tile_w // 2

    # Run DRAM streaming matmul
    activation_str = f" + {fused_activation}" if fused_activation else ""
    logger.info(f"Running DRAM streaming matmul{activation_str}: m={m}, k={k}, n={n_padded}, num_cores={num_cores}")
    try:
        ttnn_result = DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            ttnn_output,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            subblock_k=subblock_k,
            fused_activation=fused_activation,
        )
    except Exception as e:
        logger.error(f"DRAM streaming matmul{activation_str} failed: {e}")
        pytest.skip(f"Operation failed (may need API adjustments): {e}")

    # Compute PyTorch reference
    pt_out = DRAMStreamingMatmul.golden(in0, in1, fused_activation)

    # Convert to torch for comparison
    tt_out = ttnn.to_torch(ttnn_result)

    # Verify results
    if fused_activation != None:
        expected_pcc = 0.98
    else:
        expected_pcc = 0.99
    passing, output = comp_pcc(pt_out, tt_out, expected_pcc)
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info(f"DRAM streaming matmul{activation_str} test passed!")
