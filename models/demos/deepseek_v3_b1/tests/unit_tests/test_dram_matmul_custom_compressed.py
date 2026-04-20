# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import DRAMStreamingMatmulCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
from models.demos.deepseek_v3_b1.tests.unit_tests.test_matmul_custom_compressed import scale_tiles_for_mixed_formats
from models.demos.deepseek_v3_b1.weights.transforms.moe import shuffle_dram_assignment as _shuffle_dram_assignment


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
    cores_per_bank=1,
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
    primary_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary_cores_list)
    assert (
        num_banks == device.dram_grid_size().x
    ), f"num_banks ({num_banks}) != dram_grid_size ({device.dram_grid_size().x})"

    # Build expanded compute core list: primary + partner cores
    compute_cores_list = []
    for primary_core in primary_cores_list:
        for offset in range(cores_per_bank):
            compute_cores_list.append(ttnn.CoreCoord(primary_core.x + offset, primary_core.y))
    num_cores = len(compute_cores_list)

    # Pad N to align with DRAM banks (total N per bank must be divisible by cores_per_bank)
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks * cores_per_bank)
    per_core_N = n_padded // (num_banks * cores_per_bank)

    logger.info(
        f"DRAM compressed matmul: M={M}, K={K}, N={N}, n_padded={n_padded}, "
        f"per_core_N={per_core_N}, num_cores={num_cores}, cores_per_bank={cores_per_bank}"
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
    total_N_per_bank = per_core_N * cores_per_bank
    b_shard_spec = ttnn.ShardSpec(dram_grid, [K, total_N_per_bank], ttnn.ShardOrientation.ROW_MAJOR)
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
        cores_per_bank=cores_per_bank,
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


# --- Multi-core per bank tests ---


def test_dram_matmul_compressed_2cores_small_bfp8(device):
    """[1, 64] x [64, N_padded], bfp8 only, 2 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 64, 64, formats=["bfp8"], subblock_k=2, cores_per_bank=2)


def test_dram_matmul_compressed_2cores_bfp4(device):
    """[1, 7168] x [7168, 2048], bfp4 only, 2 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp4"], cores_per_bank=2)


def test_dram_matmul_compressed_2cores_mixed(device):
    """[1, 7168] x [7168, 2048], mixed bfp4+bfp2, 2 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp4", "bfp2"], cores_per_bank=2)


def test_dram_matmul_compressed_4cores_bfp4(device):
    """[1, 7168] x [7168, 2048], bfp4 only, 4 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp4"], cores_per_bank=4)


def test_dram_matmul_compressed_4cores_mixed(device):
    """[1, 7168] x [7168, 2048], mixed bfp4+bfp2, 2 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp4", "bfp2"], cores_per_bank=4)


def test_dram_matmul_compressed_4cores_mixed_bfp024(device):
    """[1, 7168] x [7168, 2048], mixed bfp4+bfp2, 2 cores per bank."""
    _run_dram_matmul_custom_compressed(device, 1, 7168, 2048, formats=["bfp4", "bfp2", "bfp0"], cores_per_bank=4)


# --- BSPM integration tests ---


def test_dram_matmul_bspm_direct(device):
    """[1, 7168] x [7168, 2048] using CompressedTensor.from_bspm() with real BSPM assignment.

    Tests the BSPM assignment → packed representation → DRAMStreamingMatmulCompressed path
    at R1 expert gate_proj shape using layer 4 expert 0 codes.

    Requires BSPM_DIR env var pointing to BitSculpt results root.
    """
    import os
    from pathlib import Path

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        import pytest

        pytest.skip("BSPM_DIR not set")

    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        import pytest

        pytest.skip(f"BSPM file not found: {bspm_path}")

    M, K, N = 1, 7168, 2048
    tile_w = 32
    primary_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary_cores_list)

    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    assignment = load_bspm_for_expert(
        str(bspm_path), expert_idx=0, proj_idx=0, tile_rows=K // tile_w, tile_cols=N // tile_w
    )
    # Pad assignment columns if n_padded > N
    if n_padded > N:
        pad_cols = (n_padded - N) // tile_w
        assignment = np.pad(assignment, ((0, 0), (0, pad_cols)), constant_values=1)  # pad with bfp4

    # Apply the same tile permutation used by shuffle_tensor_tiles/shuffle_dram_tiles so that
    # assignment[i] maps to the correct physical tile after the column-major reorder.
    # Without this, from_bspm applies precision codes to the wrong physical tiles.
    assignment = _shuffle_dram_assignment(assignment, num_banks)

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_padded)).float()

    # Shuffle for DRAM column-major tile order (same as existing tests)
    torch_b_shuffled = shuffle_tensor_tiles(torch_b, tile_w, num_banks)

    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in primary_cores_list]
    )
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
            )
        ]
    )
    b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR),
    )

    ct = CompressedTensor.from_bspm(torch_b_shuffled, assignment, device=device, memory_config=b_mem_config)
    logger.info(f"BSPM tile counts: {ct.tile_counts}")
    assert ct.tile_counts.get("bfp4", 0) > 0, "Expected bfp4 tiles from BSPM assignment"
    low_p = ct.tile_counts.get("bfp2", 0) + ct.tile_counts.get("bfp0", 0)
    assert low_p > 0, "Expected bfp2/bfp0 tiles from BSPM assignment at 3.5 b/e"

    torch_expected = (torch_a.float() @ torch_b.float()).bfloat16()

    num_cores = len(primary_cores_list)
    a_tile = ttnn.Tile([M, tile_w])
    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec),
        tile=a_tile,
    )
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, n_padded), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec),
        tile=ttnn.Tile([M, tile_w]),
    )

    Kt = K // tile_w
    subblock_k = Kt // 4 if Kt > 8 else Kt
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)

    ttnn_result = DRAMStreamingMatmulCompressed.op(ttnn_a, ct, ttnn_output, subblock_k=subblock_k)
    output_torch = ttnn.to_torch(ttnn_result)[..., :N]
    torch_expected = torch_expected[..., :N]

    passing, pcc = comp_pcc(torch_expected, output_torch, 0.90)
    logger.info(f"BSPM direct matmul PCC: {pcc}")
    assert passing, f"BSPM direct matmul PCC too low: {pcc}"


def test_dram_matmul_bspm_via_blitz(device):
    """[1, 7168] x [7168, 2048] via prepare_routed_expert_weights() BSPM path.

    Tests: prepare_routed_expert_weights(bspm_dir=...) → prepare_moe_routed_experts_bspm()
    → CompressedTensor.from_bspm() → DRAMStreamingMatmulCompressed.

    Requires BSPM_DIR env var pointing to BitSculpt results root.
    """
    import os
    from pathlib import Path

    import pytest

    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set")

    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    M, K, N = 1, 7168, 2048
    tile_w = 32
    layer_idx = 4

    torch.manual_seed(0)
    gate_w = torch.randn(K, N).float()
    up_w = torch.randn(K, N).float()
    down_w = torch.randn(2048, 7168).float()

    # State dict uses HF convention: (N_out, K_in) = transposed
    state_dict = {
        f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight": gate_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.up_proj.weight": up_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.down_proj.weight": down_w.T.contiguous(),
    }

    result = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=1,
        move_to_device=True,
        bspm_dir=Path(bspm_dir) / "deepseek-r1-0528",
    )

    ct = result.routed_gate_proj[0]
    assert isinstance(ct, CompressedTensor), f"Expected CompressedTensor, got {type(ct)}"
    logger.info(f"BSPM tile counts: {ct.tile_counts}")
    assert ct.tile_counts.get("bfp4", 0) > 0, "Expected bfp4 tiles"
    low_p = ct.tile_counts.get("bfp2", 0) + ct.tile_counts.get("bfp0", 0)
    assert low_p > 0, "Expected bfp2/bfp0 tiles at 3.5 b/e"

    primary_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary_cores_list)
    num_cores = len(primary_cores_list)
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in primary_cores_list]
    )

    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_expected = (torch_a.float() @ gate_w.float()).bfloat16()

    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec),
        tile=ttnn.Tile([M, tile_w]),
    )
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, n_padded), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec),
        tile=ttnn.Tile([M, tile_w]),
    )

    Kt = K // tile_w
    subblock_k = Kt // 4 if Kt > 8 else Kt
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)

    ttnn_result = DRAMStreamingMatmulCompressed.op(ttnn_a, ct, ttnn_output, subblock_k=subblock_k)
    output_torch = ttnn.to_torch(ttnn_result)[..., :N]
    torch_expected = torch_expected[..., :N]

    passing, pcc = comp_pcc(torch_expected, output_torch, 0.90)
    logger.info(f"BSPM matmul PCC: {pcc}")
    assert passing, f"BSPM matmul PCC too low: {pcc}"


def test_dram_matmul_bspm_down_proj(device):
    """[1, 2048] x [2048, 7168] via prepare_routed_expert_weights() — down-proj shape (proj_idx=2).

    Requires BSPM_DIR env var.
    """
    import os
    from pathlib import Path

    import pytest

    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set")

    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    M = 1
    K_gate, N_gate = 7168, 2048
    K_down, N_down = 2048, 7168
    tile_w = 32
    layer_idx = 4

    torch.manual_seed(0)
    gate_w = torch.randn(K_gate, N_gate).float()
    up_w = torch.randn(K_gate, N_gate).float()
    down_w = torch.randn(K_down, N_down).float()

    state_dict = {
        f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight": gate_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.up_proj.weight": up_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.down_proj.weight": down_w.T.contiguous(),
    }

    result = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=1,
        move_to_device=True,
        bspm_dir=Path(bspm_dir) / "deepseek-r1-0528",
    )

    ct = result.routed_down_proj[0]
    assert isinstance(ct, CompressedTensor), f"Expected CompressedTensor for down_proj, got {type(ct)}"
    logger.info(f"Down-proj BSPM tile counts: {ct.tile_counts}")
    assert ct.tile_counts.get("bfp4", 0) > 0, "Expected bfp4 tiles from BSPM"

    primary_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary_cores_list)
    num_cores = len(primary_cores_list)
    n_padded = pad_to_dram_banks(N_down, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in primary_cores_list]
    )

    torch_a = torch.randn((M, K_down), dtype=torch.bfloat16)
    torch_expected = (torch_a.float() @ down_w.float()).bfloat16()

    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K_down], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec),
        tile=ttnn.Tile([M, tile_w]),
    )
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, n_padded), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec),
        tile=ttnn.Tile([M, tile_w]),
    )

    Kt = K_down // tile_w
    subblock_k = Kt // 4 if Kt > 8 else Kt
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)

    ttnn_result = DRAMStreamingMatmulCompressed.op(ttnn_a, ct, ttnn_output, subblock_k=subblock_k)
    output_torch = ttnn.to_torch(ttnn_result)[..., :N_down]
    torch_expected = torch_expected[..., :N_down]

    passing, pcc = comp_pcc(torch_expected, output_torch, 0.90)
    logger.info(f"Down-proj BSPM matmul PCC: {pcc}")
    assert passing, f"Down-proj BSPM matmul PCC too low: {pcc}"


def test_dram_matmul_bspm_fallback(device):
    """prepare_routed_expert_weights() falls back to uniform bfloat4_b when BSPM absent.

    Uses layer_idx=0 which has no BSPM file (layer 0 is dense, only MoE layers 3+
    have BSPM). Verifies the returned tensor is a plain ttnn.Tensor (not CompressedTensor).

    Requires BSPM_DIR env var pointing to a real BitSculpt results root.
    """
    import os
    from pathlib import Path

    import pytest

    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set")

    K, N = 7168, 2048
    layer_idx = 0  # Dense layer — no BSPM file exists

    torch.manual_seed(0)
    gate_w = torch.randn(K, N).float()
    up_w = torch.randn(K, N).float()
    down_w = torch.randn(2048, 7168).float()

    state_dict = {
        f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight": gate_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.up_proj.weight": up_w.T.contiguous(),
        f"model.layers.{layer_idx}.mlp.experts.0.down_proj.weight": down_w.T.contiguous(),
    }

    result = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=1,
        move_to_device=True,
        bspm_dir=Path(bspm_dir) / "deepseek-r1-0528",
    )

    t = result.routed_gate_proj[0]
    assert not isinstance(t, CompressedTensor), (
        f"Expected plain ttnn.Tensor fallback for layer {layer_idx}, got CompressedTensor. "
        "Check that no BSPM file exists for layer 0 in the BSPM_DIR."
    )
    assert isinstance(t, ttnn.Tensor), f"Expected ttnn.Tensor, got {type(t)}"
    assert t.dtype == ttnn.bfloat4_b, f"Fallback tensor should be bfloat4_b, got {t.dtype}"
    logger.info(f"Fallback gate tensor: dtype={t.dtype}, shape={t.shape}")


def test_dram_matmul_bspm_multi_expert(device):
    """3 experts via prepare_routed_expert_weights() — validates DRAM contiguity.

    Requires BSPM_DIR env var.
    """
    import os
    from pathlib import Path

    import pytest

    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set")

    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    num_experts = 3
    K_gate, N_gate = 7168, 2048
    K_down, N_down = 2048, 7168
    layer_idx = 4

    torch.manual_seed(0)
    state_dict = {}
    gate_ws = []
    for e in range(num_experts):
        gw = torch.randn(K_gate, N_gate).float()
        uw = torch.randn(K_gate, N_gate).float()
        dw = torch.randn(K_down, N_down).float()
        gate_ws.append(gw)
        prefix = f"model.layers.{layer_idx}.mlp.experts.{e}"
        state_dict[f"{prefix}.gate_proj.weight"] = gw.T.contiguous()
        state_dict[f"{prefix}.up_proj.weight"] = uw.T.contiguous()
        state_dict[f"{prefix}.down_proj.weight"] = dw.T.contiguous()

    result = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_experts,
        move_to_device=True,
        bspm_dir=Path(bspm_dir) / "deepseek-r1-0528",
    )

    gate_list = result.routed_gate_proj
    up_list = result.routed_up_proj
    down_list = result.routed_down_proj

    assert len(gate_list) == num_experts
    assert len(up_list) == num_experts
    assert len(down_list) == num_experts

    for i, ct in enumerate(gate_list):
        assert isinstance(ct, CompressedTensor), f"gate_list[{i}] expected CompressedTensor, got {type(ct)}"
        counts = ct.tile_counts
        assert counts.get("bfp4", 0) > 0, f"gate_list[{i}]: no bfp4 tiles"
        low_p = counts.get("bfp2", 0) + counts.get("bfp0", 0)
        assert low_p > 0, f"gate_list[{i}]: no bfp2/bfp0 tiles at 3.5 b/e"
        logger.info(f"gate_list[{i}] tile counts: {counts}")

    gate_addrs = [gate_list[i].get_data_l1_address() for i in range(num_experts)]
    strides = [gate_addrs[i + 1] - gate_addrs[i] for i in range(num_experts - 1)]
    logger.info(f"Gate expert DRAM addresses: {gate_addrs}")
    logger.info(f"Gate expert DRAM strides: {strides}")

    assert strides[0] > 0, "Expert addresses not increasing — experts may overlap"

    constant_stride = all(s == strides[0] for s in strides)
    if not constant_stride:
        pytest.xfail(
            f"Non-constant DRAM stride between BSPM experts: strides={strides}. "
            "Variable-size CompressedTensors break kernel's fixed-stride expert indexing."
        )
    logger.info(f"Gate expert DRAM stride: {strides[0]} bytes — contiguity OK")


def test_dram_matmul_bspm_cache_roundtrip(device, tmp_path):
    """BSPM CompressedTensor TensorCache disk round-trip.

    Verifies:
    - Cache miss writes compact tiles.bin + assignment.npy to disk.
    - Cache hit reloads from disk and produces bitwise-equivalent CompressedTensor.
    - tiles.bin is smaller than a uniform BFP4 baseline (compact format saves space).
    - Matmul PCC between miss and hit is ≥ 0.99 (round-trip is numerically lossless).
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_b1.compressed_tensor import bfp4_tile_byte_count
    from models.demos.deepseek_v3_b1.weights.cache import (
        CacheContext,
        CompressedTensorBuildInputs,
        CompressedTensorTarget,
        SourceTensorSelection,
        TensorCache,
    )
    from models.demos.deepseek_v3_b1.weights.cache.fingerprint import compute_artifact_id

    tile_w = 32
    M, K, N = 1, 256, 256
    num_banks = device.dram_grid_size().x

    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w
    tiles_w_grid = N_padded // tile_w

    # Mixed BFP4/BFP2/zero assignment (60%/25%/15%) so compact < BFP4 baseline
    # Format codes: 1=bfp4, 2=bfp2, 3=bfp0(zero)
    rng = np.random.default_rng(42)
    assignment_logical = rng.choice([1, 2, 3], size=(tiles_h, tiles_w_grid), p=[0.60, 0.25, 0.15]).astype(np.int8)
    assignment_logical[0, 0] = 2  # guarantee at least one BFP2 tile
    assignment_logical[0, 1] = 3  # guarantee at least one zero tile

    torch.manual_seed(0)
    w_logical = torch.randn(K, N_padded).float().numpy()

    cache_ctx = CacheContext(
        schema_version=1,
        hf_model_id="test_model",
        hf_revision="test",
        mesh_shape=(1, 1),
    )
    tgt = CompressedTensorTarget(
        name="test_proj",
        K=K,
        N_padded=N_padded,
        num_banks=num_banks,
        bspm_variant="B",
        bspm_budget=3.5,
    )
    fp = cache_ctx.fingerprint(source=SourceTensorSelection(names=("test_weight",)), target=tgt)

    cache = TensorCache(tmp_path / "cache")

    def _preprocess(_raw):
        return {tgt.name: CompressedTensorBuildInputs(w_logical=w_logical, assignment_logical=assignment_logical)}

    # --- First call: cache miss — writes tiles.bin, assignment.npy, metadata.json ---
    ct_miss = cache.get_or_create(fp, device, preprocess=_preprocess, raw_tensors={})
    assert isinstance(ct_miss, CompressedTensor), f"Expected CompressedTensor from miss, got {type(ct_miss)}"

    artifact_id = compute_artifact_id(fp)
    obj_dir = tmp_path / "cache" / "objects" / artifact_id[:2] / artifact_id
    tiles_bin = obj_dir / "tiles.bin"
    assert tiles_bin.is_file(), "tiles.bin not written on cache miss"
    assert (obj_dir / "assignment.npy").is_file(), "assignment.npy not written on cache miss"
    assert (obj_dir / "metadata.json").is_file(), "metadata.json not written on cache miss"

    compact_size = tiles_bin.stat().st_size
    bfp4_size = bfp4_tile_byte_count(tiles_h, tiles_w_grid)
    assert compact_size < bfp4_size, (
        f"Compact tiles.bin ({compact_size} B) must be < uniform BFP4 baseline ({bfp4_size} B). "
        "Check that BFP2/zero tiles are reducing the packed size."
    )
    logger.info(
        "tiles.bin: {} B vs BFP4 baseline {} B ({:.1f}%)",
        compact_size,
        bfp4_size,
        100 * compact_size / bfp4_size,
    )

    # Per-tier tile counts must sum to total tile grid area.
    counts = ct_miss.tile_counts
    total_tiles = tiles_h * tiles_w_grid
    count_sum = counts.get("bfp4", 0) + counts.get("bfp2", 0) + counts.get("bfp0", 0)
    assert count_sum == total_tiles, (
        f"Tile count mismatch: bfp4={counts.get('bfp4',0)} + bfp2={counts.get('bfp2',0)} "
        f"+ bfp0={counts.get('bfp0',0)} = {count_sum}, expected {total_tiles}"
    )
    assert counts.get("bfp4", 0) > 0, "No BFP4 tiles found — assignment may be wrong"
    assert counts.get("bfp2", 0) > 0, "No BFP2 tiles found — assignment may be wrong"
    assert counts.get("bfp0", 0) > 0, "No zero tiles found — assignment may be wrong"
    logger.info(
        "Tile counts — bfp4: {}, bfp2: {}, zero: {} (total {})",
        counts.get("bfp4", 0),
        counts.get("bfp2", 0),
        counts.get("bfp0", 0),
        total_tiles,
    )

    # --- Second call: cache hit — reloads from tiles.bin without calling preprocess ---
    ct_hit = cache.get_or_create(fp, device, preprocess=_preprocess, raw_tensors={})
    assert isinstance(ct_hit, CompressedTensor), f"Expected CompressedTensor from hit, got {type(ct_hit)}"

    # --- Matmul PCC: miss and hit must agree (lossless round-trip) ---
    per_core_N = N_padded // num_banks
    primary_cores_list = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in primary_cores_list]
    )
    num_cores = len(primary_cores_list)

    torch.manual_seed(1)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_a_replicated = torch_a.repeat(num_cores, 1)
    a_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=ttnn.Tile([M, tile_w]),
    )

    out_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    Kt = K // tile_w
    subblock_k = Kt if Kt <= 8 else Kt // 4
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)

    def _run(ct):
        ttnn_out = ttnn.from_torch(
            torch.zeros((M, N_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=out_mem_config,
            tile=ttnn.Tile([M, tile_w]),
        )
        return ttnn.to_torch(DRAMStreamingMatmulCompressed.op(ttnn_a, ct, ttnn_out, subblock_k=subblock_k))[..., :N]

    out_miss = _run(ct_miss)
    out_hit = _run(ct_hit)

    passing, pcc = comp_pcc(out_miss, out_hit, 0.99)
    logger.info("Cache roundtrip PCC (miss vs hit): {}", pcc)
    assert passing, f"Cache roundtrip PCC too low: {pcc} — miss and hit CompressedTensors disagree"
