# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DRAM Streaming Matmul with Compressed Weights - Tests

Tests the combined DRAM streaming + compressed tensor matmul where:
- Input A is REPLICATED on compute cores (each core has full [M, K])
- Input B (compressed weights) is WIDTH_SHARDED in DRAM with mixed BFP formats
- Output is WIDTH_SHARDED in L1 on compute cores

Key difference from test_matmul_custom_compressed.py (L1 version):
B lives in DRAM and is streamed in variable-size subblocks.
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import DRAMStreamingMatmulCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def pad_to_dram_banks(num, tile_w, lcm):
    """Pad number to be aligned with DRAM banks."""
    remainder = num % lcm
    if remainder == 0:
        return num
    return num + (lcm - remainder)


def _run_dram_matmul_custom_compressed(
    device,
    M,
    K,
    N,
    formats,
    subblock_k=None,
    threshold=0.993,
    pcc_threshold=0.98,
):
    """Helper: run DRAM streaming compressed A @ decompress(B_compressed).

    B [K, N] is WIDTH_SHARDED in DRAM across DRAM banks.
    A [M, K] is replicated on compute cores.
    Output [M, N] is WIDTH_SHARDED on compute cores.
    """
    tile_w = 32

    # Get DRAM bank grid and compute cores
    compute_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores_list)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks, f"num_cores ({num_cores}) != num_banks ({num_banks})"

    # Pad N to align with DRAM banks
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    logger.info(
        f"DRAM compressed matmul: M={M}, K={K}, N={N}, n_padded={n_padded}, "
        f"per_core_N={per_core_N}, num_cores={num_cores}"
    )

    Kt = K // tile_w
    if subblock_k is None:
        # Default: use K/4 for large K, full K for small K
        if Kt > 8:
            subblock_k = Kt // 4
        else:
            subblock_k = Kt
    # Ensure subblock_k is even
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)
    assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
    num_subblocks_k = Kt // subblock_k

    logger.info(f"Kt={Kt}, subblock_k={subblock_k}, num_subblocks_k={num_subblocks_k}")

    # Build CoreRangeSet for compute cores
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores_list]
    )

    # Create test data
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_padded)).float()
    scale_tiles_for_mixed_formats(torch_b, formats)

    # Shuffle torch_b to column-major tile order within each DRAM shard.
    # DRAM streaming reads K tiles contiguously per N column, but ttnn stores
    # tiles row-major. Pre-shuffling ensures the compressed data in DRAM is in
    # the order the kernel expects.
    torch_b_shuffled = shuffle_tensor_tiles(torch_b, tile_w, num_banks)

    # Create CompressedTensor in DRAM from the shuffled tensor
    bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae)

    # DRAM bank grid for B tensor
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        ]
    )
    b_shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, b_shard_spec)

    ct = CompressedTensor.from_torch(torch_b_shuffled, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"DRAM compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Verify all requested formats are used
    counts = ct.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    # Golden: A @ B (original unshuffled, unquantized)
    # PCC threshold accounts for BFP quantization error.
    torch_expected = (torch_a.float() @ torch_b.float()).bfloat16()

    # Input A — replicated: HEIGHT_SHARD so each compute core gets [M, K]
    a_tile = ttnn.Tile([M, tile_w])
    torch_a_replicated = torch_a.repeat(num_cores, 1)  # [M*num_cores, K]
    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    # Output — WIDTH_SHARDED on compute cores
    out_tile = ttnn.Tile([M, tile_w])
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, n_padded), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    # Run DRAM streaming compressed matmul
    ttnn_result = DRAMStreamingMatmulCompressed.op(
        ttnn_a,
        ct,
        ttnn_output,
        subblock_k=subblock_k,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    # Slice to original N if padded
    if n_padded != N:
        output_torch = output_torch[..., :N]
        torch_expected = torch_expected[..., :N]

    assert (
        output_torch.shape == torch_expected.shape
    ), f"Shape mismatch: got {output_torch.shape}, expected {torch_expected.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)
    assert passing, pcc_message


# --- Basic tests ---


def test_dram_matmul_compressed_small_bfp8(device):
    """[1, 64] x [64, N_padded], bfp8 only. K=2 tiles."""
    _run_dram_matmul_custom_compressed(device, 1, 64, 64, formats=["bfp8"], subblock_k=2)


def test_dram_matmul_compressed_multi_n_bfp8(device):
    """[1, 128] x [128, N_padded], bfp8 only. Kt=4, single subblock, multi-N columns."""
    _run_dram_matmul_custom_compressed(device, 1, 128, 256, formats=["bfp8"], subblock_k=4)


def test_dram_matmul_compressed_multi_subblock_bfp8(device):
    """[1, 128] x [128, 64], bfp8 only. Kt=4, subblock_k=2, 2 subblocks, single N column."""
    _run_dram_matmul_custom_compressed(device, 1, 128, 256, formats=["bfp8"], subblock_k=2)


def test_dram_matmul_compressed_bfp8(device):
    """[1, 7168] x [7168, 2048], bfp8 only. DeepSeek shape."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp8"])


def test_dram_matmul_compressed_mixed(device):
    """[1, 7168] x [7168, 2048], mixed bfp8+bfp4."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp8", "bfp4"])


def test_dram_matmul_compressed_reversed_shape(device):
    """[1, 2048] x [2048, 7168], bfp8+bfp4. Transposed DeepSeek shape."""
    _run_dram_matmul_custom_compressed(device, 1, 2048, 7168, formats=["bfp8", "bfp4"])
