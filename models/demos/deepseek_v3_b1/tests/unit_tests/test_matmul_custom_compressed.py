# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Custom Matmul with Compressed Weights - Single Core

Tiles are pre-sorted by format. The kernel reconfigures the unpacker
only once per format group instead of per tile.
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed


def scale_tiles_for_mixed_formats(b_torch, formats):
    """Adjust tiles so the assigner picks different formats.

    bfp8 tiles: high within-block dynamic range (elements span orders of magnitude)
    bfp4 tiles: moderate range (randn, shared exponent works ok for bfp4)
    bfp2 tiles: uniform values within each block (shared exponent covers all)
    """
    if len(formats) <= 1:
        return

    M, N = b_torch.shape
    num_fmts = len(formats)
    tiles_h, tiles_w = M // 32, N // 32

    for idx in range(tiles_h * tiles_w):
        tr, tc = idx // tiles_w, idx % tiles_w
        fmt = formats[idx % num_fmts]
        r0, r1 = tr * 32, (tr + 1) * 32
        c0, c1 = tc * 32, (tc + 1) * 32
        tile = b_torch[r0:r1, c0:c1]
        if fmt == "bfp8":
            # High within-block dynamic range: multiply each row by exponentially
            # increasing factors. Elements in the same 16-element block will span
            # several orders of magnitude, requiring bfp8's 7 mantissa bits.
            for r in range(32):
                tile[r, :] *= 2.0 ** (r % 16)
        elif fmt == "bfp2":
            # bfp2 = 1 mantissa bit + shared exponent per 16-element block.
            # Make each row a random power of 2 (shared exp) with random signs.
            # Values like ±4, ±0.25, ±16 etc — varied across rows but uniform within block.
            for r in range(32):
                exp = torch.randint(-3, 4, (1,)).float()
                signs = torch.sign(torch.randn(32))
                signs[signs == 0] = 1.0
                b_torch[r0 + r, c0:c1] = signs * (2.0**exp)
        elif fmt == "bfp0":
            # bfp0 = zero tile. Use tiny random values that round to zero
            # under the bfp0_mae_threshold (1e-3).
            b_torch[r0:r1, c0:c1] = torch.randn(32, 32) * 1e-3
        # bfp4: keep randn as-is


def _run_matmul_custom_compressed(
    device,
    M,
    K,
    N,
    formats,
    num_cores=1,
    threshold=0.993,
    pcc_threshold=0.98,
    impl="constexpr_compact",
    tile_scaler=None,
):
    """Helper: run custom compressed A @ decompress(B_compressed).

    B [K, N] is width-sharded across num_cores (each core gets [K, N/num_cores]).
    A [M, K] is replicated on every core via HEIGHT_SHARDED.
    Output [M, N] is width-sharded.
    """
    assert N % (num_cores * 32) == 0, f"N={N} must be divisible by num_cores*32={num_cores * 32}"
    n_per_core = N // num_cores

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()
    if tile_scaler is not None:
        tile_scaler(torch_b, formats)
    else:
        scale_tiles_for_mixed_formats(torch_b, formats)

    # Core grid: fill full rows, remainder on the last row
    max_cols = device.compute_with_storage_grid_size().x
    max_rows = device.compute_with_storage_grid_size().y
    full_rows = num_cores // max_cols
    remainder = num_cores % max_cols
    assert (
        full_rows + (1 if remainder else 0) <= max_rows
    ), f"num_cores={num_cores} exceeds device grid {max_cols}x{max_rows}"
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_cols - 1, full_rows - 1)))
    if remainder > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    core_grid = ttnn.CoreRangeSet(ranges)

    # Compress B — width-sharded across cores
    bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae)
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    ct = CompressedTensor.from_torch(torch_b, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"Custom compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")
    assignment = ct.get_assignment()
    flat = assignment.ravel()
    # Count format runs
    runs = []
    i = 0
    while i < len(flat):
        fmt = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == fmt:
            count += 1
        runs.append((int(fmt), count))
        i += count
    logger.info(f"Assignment: {len(flat)} tiles, {len(runs)} runs, first 10 runs: {runs[:10]}")

    # Verify all requested formats are used
    counts = ct.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    # Golden: A @ B (original float)
    torch_expected = (torch_a.float() @ torch_b).bfloat16()

    # A — replicated: stack num_cores copies, height-shard so each core gets [M, K]
    torch_a_replicated = torch_a.repeat(num_cores, 1)  # [M*num_cores, K]
    a_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    # Output — width-sharded
    out_tile = ttnn.Tile([M, 32])
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    # Run custom compressed matmul
    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output, impl=impl)

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)
    assert passing, pcc_message


# --- Constexpr compact path ---


def test_matmul_custom_compressed_small(device):
    """[1, 64] x [64, 32], bfp8 only. K=2 tiles."""
    _run_matmul_custom_compressed(device, 1, 64, 32, formats=["bfp8"])


def test_matmul_custom_compressed_mixed(device):
    """[1, 256] x [256, 32], mixed bfp8+bfp4. K=8 tiles."""
    _run_matmul_custom_compressed(device, 1, 256, 32, formats=["bfp8", "bfp4"])


def test_matmul_custom_compressed_large(device):
    """[1, 7168] x [7168, 32], mixed bfp8+bfp4. DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4"])


def test_matmul_custom_compressed_large_uniform(device):
    """[1, 7168] x [7168, 32], bfp8 only. DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8"])


def test_matmul_custom_compressed_wide(device):
    """[1, 64] x [64, 64], bfp8. out_w=2."""
    _run_matmul_custom_compressed(device, 1, 64, 64, formats=["bfp8"])


def test_matmul_custom_compressed_wide_mixed(device):
    """[1, 256] x [256, 128], mixed bfp8+bfp4. out_w=4."""
    _run_matmul_custom_compressed(device, 1, 256, 128, formats=["bfp8", "bfp4"])


def test_matmul_custom_compressed_multicore_2cores(device):
    """[1, 7168] x [7168, 128], bfp8, 2 cores."""
    _run_matmul_custom_compressed(device, 1, 7168, 128, formats=["bfp8"], num_cores=2)


def test_matmul_custom_compressed_multicore_mixed_13cores(device):
    """[1, 7168] x [7168, 416], mixed bfp8+bfp4, 13 cores."""
    _run_matmul_custom_compressed(device, 1, 7168, 32 * 13, formats=["bfp8", "bfp4"], num_cores=13)


def test_matmul_custom_compressed_multicore_mixed_32cores(device):
    """[1, 7168] x [7168, 2048], mixed bfp8+bfp4, 32 cores."""
    _run_matmul_custom_compressed(device, 1, 7168, 64 * 32, formats=["bfp8", "bfp4"], num_cores=32)


# --- Full constexpr unroll path ---


def test_matmul_custom_compressed_constexpr_unroll_mixed(device):
    """[1, 256] x [256, 32], mixed bfp8+bfp4 using fully unrolled constexpr path."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4"], impl="constexpr_unroll")


# --- Runtime path (no constexpr unroll) ---


def test_matmul_custom_compressed_runtime_large(device):
    """[1, 7168] x [7168, 32], bfp8. Runtime path, DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4"], impl="runtime")


def test_matmul_custom_compressed_runtime_large_uniform(device):
    """[1, 7168] x [7168, 32], bfp8. Runtime path, DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8"], impl="runtime")


def test_matmul_custom_compressed_runtime_uniform_wide(device):
    """[1, 512] x [512, 256], bfp8. Runtime path, DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 512, 256, formats=["bfp8"], impl="runtime")


def test_matmul_custom_compressed_runtime_wide(device):
    """[1, 512] x [512, 256], bfp8. Runtime path, DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 512, 256, formats=["bfp8", "bfp4"], impl="runtime")


def test_matmul_custom_compressed_mixed_with_bfp2_runtime(device):
    """[1, 7168] x [7168, 32], mixed bfp8+bfp4+bfp2 via runtime impl."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4", "bfp2"], impl="runtime")


def test_matmul_custom_compressed_mixed_with_bfp20_runtime(device):
    """[1, 7168] x [7168, 32], mixed bfp8+bfp4+bfp2+bfp0 via runtime impl."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4", "bfp2", "bfp0"], impl="runtime")


def test_matmul_custom_compressed_mixed_with_bfp20_wide_runtime(device):
    """[1, 7168] x [7168, 256], mixed bfp8+bfp4+bfp2+bfp0 via runtime impl."""
    _run_matmul_custom_compressed(device, 1, 7168, 256, formats=["bfp8", "bfp4", "bfp2", "bfp0"], impl="runtime")


def scale_tiles_clustered(b_torch, formats):
    """Top half tiles get first format, bottom half get second format.
    Results in exactly 2 format runs (or 1 if only 1 format)."""
    if len(formats) <= 1:
        return
    M, N = b_torch.shape
    tiles_h = M // 32
    tiles_w = N // 32
    half = tiles_h // 2
    for tr in range(tiles_h):
        fmt = formats[0] if tr < half else formats[1]
        for tc in range(tiles_w):
            r0, r1 = tr * 32, (tr + 1) * 32
            c0, c1 = tc * 32, (tc + 1) * 32
            tile = b_torch[r0:r1, c0:c1]
            if fmt == "bfp8":
                tile *= torch.tensor([1e-3, 1e3] * 16).unsqueeze(1)
            elif fmt == "bfp4":
                pass
            elif fmt == "bfp2":
                tile.fill_(1.0)
                tile += torch.randn_like(tile) * 0.01


def test_matmul_custom_compressed_clustered(device):
    """[1, 7168] x [7168, 32], top half bfp8, bottom half bfp4. Exactly 2 runs."""
    import numpy as np

    M, K, N = 1, 7168, 32
    n_per_core = N

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32  # 224
    tiles_w = N // 32  # 1
    half = tiles_h // 2  # 112

    # Force assignment: top half = bfp8 (index 0), bottom half = bfp4 (index 1)
    assignment = np.zeros((tiles_h, tiles_w), dtype=np.int8)
    assignment[half:, :] = 1  # bfp4

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)

    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor

    ct = CompressedTensor(torch_b, assignment, device=device, memory_config=b_mem_config)

    logger.info(f"Clustered compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")
    flat = assignment.ravel()
    runs = []
    i = 0
    while i < len(flat):
        fmt = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == fmt:
            count += 1
        runs.append((int(fmt), count))
        i += count
    logger.info(f"Assignment: {len(flat)} tiles, {len(runs)} runs: {runs}")

    torch_expected = (torch_a.float() @ ct.to_torch().float()).bfloat16()

    a_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    out_tile = ttnn.Tile([M, 32])
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed

    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output, impl="constexpr_compact")

    output_torch = ttnn.to_torch(ttnn_result)
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(pcc_message)
    assert passing, pcc_message


# ---------------------------------------------------------------------------
# BSPM real-assignment tests
# ---------------------------------------------------------------------------


def _run_matmul_custom_compressed_bspm(
    device, M, K, N, bspm_path, proj_idx, expert_idx=0, num_cores=1, pcc_threshold=0.90
):
    """Like _run_matmul_custom_compressed but uses a real BSPM assignment instead of an assigner.

    Loads the assignment for one expert projection from the binary .bspm file and calls
    CompressedTensor.from_bspm() to bypass the assigner entirely.  Random weights are used
    so the test does not require a checkpoint; the purpose is to validate that the kernel
    correctly decompresses a real mixed-precision {bfp4, bfp2, zero} tile map.
    """

    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    tile_w = 32
    assert N % (num_cores * tile_w) == 0, f"N={N} must be divisible by num_cores*tile_w={num_cores * tile_w}"
    n_per_core = N // num_cores

    assignment = load_bspm_for_expert(
        str(bspm_path),
        expert_idx=expert_idx,
        proj_idx=proj_idx,
        tile_rows=K // tile_w,
        tile_cols=N // tile_w,
    )

    torch.manual_seed(0)
    torch_b = torch.randn((K, N)).float()

    max_cols = device.compute_with_storage_grid_size().x
    full_rows = num_cores // max_cols
    remainder = num_cores % max_cols
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_cols - 1, full_rows - 1)))
    if remainder > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    core_grid = ttnn.CoreRangeSet(ranges)

    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    ct = CompressedTensor.from_bspm(torch_b, assignment, device=device, memory_config=b_mem_config)

    logger.info(f"BSPM tile counts (proj_idx={proj_idx}): {ct.tile_counts}")
    assert ct.tile_counts.get("bfp4", 0) > 0, "Expected bfp4 tiles from BSPM assignment"
    low_p = ct.tile_counts.get("bfp2", 0) + ct.tile_counts.get("bfp0", 0)
    assert low_p > 0, "Expected bfp2/bfp0 tiles at 3.5 b/e budget"

    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_expected = (torch_a.float() @ torch_b.float()).bfloat16()

    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=ttnn.Tile([M, tile_w]),
    )

    out_shard_spec = ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=ttnn.Tile([M, tile_w]),
    )

    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output)
    output_torch = ttnn.to_torch(ttnn_result)

    passing, pcc = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(f"BSPM matmul PCC: {pcc}")
    assert passing, f"BSPM matmul PCC too low: {pcc}"


def test_matmul_custom_compressed_bspm_gate_proj(device):
    """[1, 7168] x [7168, 2048] with real BSPM assignment at 3.5 b/e — gate_proj shape.

    Uses 32 cores so each shard is [7168, 64] (2 tile columns). Assignment loaded from the
    binary .bspm file for layer 4 expert 0 proj_idx=0 (gate_proj).
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    import pytest

    bspm_results_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_results_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_results_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    _run_matmul_custom_compressed_bspm(device, M=1, K=7168, N=2048, bspm_path=bspm_path, proj_idx=0, num_cores=32)


def test_matmul_custom_compressed_bspm_down_proj(device):
    """[1, 2048] x [2048, 7168] with real BSPM assignment at 3.5 b/e — down_proj shape.

    Uses 32 cores so each shard is [2048, 224] (7 tile columns). Assignment loaded from
    layer 4 expert 0 proj_idx=2 (down_proj).
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    import pytest

    bspm_results_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_results_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_results_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    _run_matmul_custom_compressed_bspm(device, M=1, K=2048, N=7168, bspm_path=bspm_path, proj_idx=2, num_cores=32)


def test_matmul_custom_compressed_bspm_mixed_budget(device):
    """Compare PCC of real 3.5 b/e BSPM vs synthetic uniform BFP4 on the same random weight.

    BFP4 (ttnn code 1) is lossless compared to BFP2/zero tiles, so uniform BFP4 must achieve
    equal or higher PCC than the mixed 3.5 b/e assignment.  Both must pass PCC >= 0.85.
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    import numpy as np
    import pytest

    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    bspm_results_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_results_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_results_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    M, K, N = 1, 7168, 2048
    tile_w = 32
    num_cores = 32
    n_per_core = N // num_cores

    assignment_35 = load_bspm_for_expert(
        str(bspm_path), expert_idx=0, proj_idx=0, tile_rows=K // tile_w, tile_cols=N // tile_w
    )
    # Synthetic uniform BFP4 baseline: ttnn code 1 = bfp4 (BS code 2 remapped via 3 - bs)
    assignment_bfp4 = np.ones((K // tile_w, N // tile_w), dtype=np.int8)

    torch.manual_seed(0)
    torch_b = torch.randn((K, N)).float()
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_expected = (torch_a.float() @ torch_b.float()).bfloat16()

    max_cols = device.compute_with_storage_grid_size().x
    full_rows = num_cores // max_cols
    remainder = num_cores % max_cols
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_cols - 1, full_rows - 1)))
    if remainder > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    core_grid = ttnn.CoreRangeSet(ranges)

    def _run_with_assignment(assignment):
        b_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR),
        )
        ct = CompressedTensor.from_bspm(torch_b, assignment, device=device, memory_config=b_mem)
        a_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR),
        )
        ttnn_a = ttnn.from_torch(
            torch_a.repeat(num_cores, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=a_mem,
            tile=ttnn.Tile([M, tile_w]),
        )
        out_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR),
        )
        ttnn_out = ttnn.from_torch(
            torch.zeros((M, N), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=out_mem,
            tile=ttnn.Tile([M, tile_w]),
        )
        return ttnn.to_torch(MatmulCustomCompressed.op(ttnn_a, ct, ttnn_out))

    out_35 = _run_with_assignment(assignment_35)
    out_bfp4 = _run_with_assignment(assignment_bfp4)

    passing_35, pcc_35 = comp_pcc(torch_expected, out_35, 0.85)
    passing_bfp4, pcc_bfp4 = comp_pcc(torch_expected, out_bfp4, 0.85)

    logger.info(f"3.5 b/e BSPM PCC: {pcc_35}")
    logger.info(f"Uniform BFP4 PCC: {pcc_bfp4}")

    assert passing_35, f"3.5 b/e BSPM PCC too low: {pcc_35}"
    assert passing_bfp4, f"Uniform BFP4 PCC too low: {pcc_bfp4}"
    # BFP4-only must be at least as good as the mixed 3.5 b/e (allow 1% tolerance for rng variance)
    assert float(pcc_bfp4) >= float(pcc_35) - 0.01, f"Uniform BFP4 PCC ({pcc_bfp4}) should be >= 3.5 b/e PCC ({pcc_35})"


def _run_matmul_interleaved_by_n(device, interleave_n, impl="constexpr_compact"):
    """[1, 7168] x [7168, 32], alternating every N tiles: bfp8*N, bfp4*N, ..."""
    import numpy as np

    M, K, N = 1, 7168, 32
    n_per_core = N

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32  # 224
    tiles_w = N // 32  # 1

    assignment = np.zeros((tiles_h, tiles_w), dtype=np.int8)
    for tr in range(tiles_h):
        assignment[tr, :] = (tr // interleave_n) % 2

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)

    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor

    ct = CompressedTensor(torch_b, assignment, device=device, memory_config=b_mem_config)

    logger.info(f"Interleaved-by-{interleave_n} compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")
    flat = assignment.ravel()
    runs = []
    i = 0
    while i < len(flat):
        fmt = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == fmt:
            count += 1
        runs.append((int(fmt), count))
        i += count
    logger.info(f"Assignment: {len(flat)} tiles, {len(runs)} runs, first 10: {runs[:10]}")

    torch_expected = (torch_a.float() @ ct.to_torch().float()).bfloat16()

    a_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    out_tile = ttnn.Tile([M, 32])
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed

    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output, impl=impl)

    output_torch = ttnn.to_torch(ttnn_result)
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(pcc_message)
    assert passing, pcc_message


def test_matmul_custom_compressed_interleaved_by2(device):
    """Alternating every 2 tiles: 112 runs of 2."""
    _run_matmul_interleaved_by_n(device, 2)


def test_matmul_custom_compressed_interleaved_by4(device):
    """Alternating every 4 tiles: 56 runs of 4."""
    _run_matmul_interleaved_by_n(device, 4)


def test_matmul_custom_compressed_interleaved_by8(device):
    """Alternating every 8 tiles: 28 runs of 8."""
    _run_matmul_interleaved_by_n(device, 8)


def test_matmul_custom_compressed_interleaved_by16(device):
    """Alternating every 16 tiles: 14 runs of 16."""
    _run_matmul_interleaved_by_n(device, 16)


def test_matmul_custom_compressed_interleaved_by32(device):
    """Alternating every 32 tiles: 7 runs of 32."""
    _run_matmul_interleaved_by_n(device, 32)


def test_matmul_custom_compressed_hybrid(device):
    """Mixed interleaving: sections of by-2, by-4, by-8, by-16 in one tensor."""
    import numpy as np

    M, K, N = 1, 7168, 32
    n_per_core = N

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32  # 224
    tiles_w = N // 32  # 1

    # Build hybrid pattern:
    # tiles 0-31:   interleave by 2  (16 runs of 2)
    # tiles 32-63:  interleave by 4  (8 runs of 4)
    # tiles 64-127: interleave by 8  (8 runs of 8)
    # tiles 128-223: interleave by 16 (6 runs of 16)
    assignment = np.zeros((tiles_h, tiles_w), dtype=np.int8)
    for tr in range(tiles_h):
        if tr < 32:
            assignment[tr, :] = (tr // 2) % 2
        elif tr < 64:
            assignment[tr, :] = ((tr - 32) // 4) % 2
        elif tr < 128:
            assignment[tr, :] = ((tr - 64) // 8) % 2
        else:
            assignment[tr, :] = ((tr - 128) // 16) % 2

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)

    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor

    ct = CompressedTensor(torch_b, assignment, device=device, memory_config=b_mem_config)

    logger.info(f"Hybrid compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")
    flat = assignment.ravel()
    runs = []
    i = 0
    while i < len(flat):
        fmt = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == fmt:
            count += 1
        runs.append((int(fmt), count))
        i += count
    logger.info(f"Assignment: {len(flat)} tiles, {len(runs)} runs, first 20: {runs[:20]}")

    torch_expected = (torch_a.float() @ ct.to_torch().float()).bfloat16()

    a_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    out_tile = ttnn.Tile([M, 32])
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed

    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output, impl="constexpr_compact")

    output_torch = ttnn.to_torch(ttnn_result)
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(pcc_message)
    assert passing, pcc_message
