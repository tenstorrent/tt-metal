# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Custom Matmul with Compressed Weights - Single Core

Tiles are pre-sorted by format. The kernel reconfigures the unpacker
only once per format group instead of per tile.
"""

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import ttnn_quantize_fn
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def _scale_tiles_clustered(b_torch, formats):
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


def _run_matmul_custom_compressed_with_assignment(
    device,
    M,
    K,
    N,
    impl,
    torch_a,
    torch_b,
    assignment,
    num_cores=1,
    pcc_threshold=0.98,
):
    """Helper: run custom compressed A @ decompress(B_compressed).

    B [K, N] is width-sharded across num_cores (each core gets [K, N/num_cores]).
    A [M, K] is replicated on every core via HEIGHT_SHARDED.
    Output [M, N] is width-sharded.
    """
    assert N % (num_cores * 32) == 0, f"N={N} must be divisible by num_cores*32={num_cores * 32}"
    assert torch_a.shape == (M, K), f"Expected torch_a shape ({M}, {K}), got {torch_a.shape}"
    assert torch_b.shape == (K, N), f"Expected torch_b shape ({K}, {N}), got {torch_b.shape}"
    assert assignment.shape == (
        K // 32,
        N // 32,
    ), f"Expected assignment shape ({K//32}, {N//32}), got {assignment.shape}"
    n_per_core = N // num_cores

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

    b_shard_spec = ttnn.ShardSpec(core_grid, [K, n_per_core], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    ct = CompressedTensor(torch_b, assignment, device=device, memory_config=b_mem_config)

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

    # Golden: A @ B (original float)
    torch_expected = (torch_a.float() @ torch_b).bfloat16()

    # A — replicated: stack num_cores copies, height-shard so each core gets [M, K]
    torch_a_replicated = torch_a.repeat(num_cores, 1)
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


def _run_matmul_custom_compressed(
    device,
    M,
    K,
    N,
    impl,
    formats,
    num_cores=1,
    pcc_threshold=0.98,
    threshold=0.993,
    tile_scaler=None,
):
    """Automatic format assignment."""

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()
    if tile_scaler is not None:
        tile_scaler(torch_b, formats)
    else:
        scale_tiles_for_mixed_formats(torch_b, formats)

    # Compress B — width-sharded across cores
    bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae)
    result = assigner.assign(torch_b, ttnn_quantize_fn)

    counts = result.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    _run_matmul_custom_compressed_with_assignment(
        device, M, K, N, impl, torch_a, torch_b, result.assignment, num_cores, pcc_threshold
    )


def _run_matmul_custom_compressed_clustered(
    device,
    M,
    K,
    N,
    impl,
    num_cores=1,
    pcc_threshold=0.98,
):
    """Top half bfp8, bottom half bfp4. Exactly 2 runs."""

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32
    tiles_w = N // 32
    half = tiles_h // 2

    # Force assignment: top half = bfp8 (index 0), bottom half = bfp4 (index 1)
    assignment = np.zeros((tiles_h, tiles_w), dtype=np.int8)
    assignment[half:, :] = 1  # bfp4

    _run_matmul_custom_compressed_with_assignment(
        device, M, K, N, impl, torch_a, torch_b, assignment, num_cores, pcc_threshold
    )


def _run_matmul_custom_compressed_interleaved_by_n(
    device,
    M,
    K,
    N,
    impl,
    interleave_n,
    num_cores=1,
    pcc_threshold=0.98,
):
    """Alternating every N tiles: bfp8*N, bfp4*N, ..."""

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32
    tiles_w = N // 32

    assignment = np.zeros((tiles_h, tiles_w), dtype=np.int8)
    for tr in range(tiles_h):
        assignment[tr, :] = (tr // interleave_n) % 2

    _run_matmul_custom_compressed_with_assignment(
        device, M, K, N, impl, torch_a, torch_b, assignment, num_cores, pcc_threshold
    )


def _run_matmul_custom_compressed_hybrid(
    device,
    M,
    K,
    N,
    impl,
    num_cores=1,
    pcc_threshold=0.98,
):
    """Mixed interleaving: sections of by-2, by-4, by-8, by-16 in one tensor."""

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()

    tiles_h = K // 32
    tiles_w = N // 32

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

    _run_matmul_custom_compressed_with_assignment(
        device, M, K, N, impl, torch_a, torch_b, assignment, num_cores, pcc_threshold
    )


SHAPES = [
    (1, 64, 32),
    (1, 64, 64),
    (1, 256, 32),
    (1, 256, 128),
    (1, 512, 256),
    (1, 7168, 32),
    (1, 7168, 64),
    (1, 7168, 256),
]

FORMATS = [
    ["bfp8"],
    ["bfp8", "bfp4"],
    ["bfp8", "bfp4", "bfp2"],
    ["bfp8", "bfp4", "bfp2", "bfp0"],
]

IMPLEMENTATIONS = [
    "constexpr_unroll",
    "constexpr_compact",
    "runtime",
    "constexpr_unroll barrier",
    "constexpr_compact barrier",
    "runtime barrier",
]


@pytest.mark.parametrize(
    "M,K,N,formats,impl",
    [
        (M, K, N, formats, impl)
        for M, K, N in SHAPES
        for formats in FORMATS
        for impl in IMPLEMENTATIONS
        if not (
            ("bfp2" in formats or "bfp0" in formats) and "barrier" not in impl
        )  # bfp2/bfp0 requires barrier synchronization
        if not ("bfp0" in formats and "constexpr" in impl)  # bfp0 does not work with constexpr
        if not (
            ((K // 32) * (N // 32)) <= 8 and len(formats) > 2
        )  # Small shapes with many formats may not have enough tiles
        if not (M == 1 and K == 7168 and N == 256 and formats == ["bfp8"])  # Not enough memory
        if not (M == 1 and K == 7168 and N == 256 and "constexpr" in impl)  # Not enough program memory
    ],
)
def test_matmul_custom_compressed_single_core(device, M, K, N, formats, impl):
    """Single core tests with automatic format assignment across all implementations."""
    if (
        (M, K, N) == (1, 512, 256)
        and formats != ["bfp8"]
        and impl in ("constexpr_unroll barrier", "constexpr_compact barrier")
    ):
        pytest.skip("FIXME: PCC ERROR")
    _run_matmul_custom_compressed(device, M, K, N, impl=impl, formats=formats)


@pytest.mark.parametrize(
    "M,K,N_per_core,formats,num_cores,impl",
    [
        (M, K, N_per_core, formats, num_cores, impl)
        for M, K, N_per_core in SHAPES
        for formats in FORMATS
        for num_cores in [2, 13, 32]
        for impl in IMPLEMENTATIONS
        if not (
            ("bfp2" in formats or "bfp0" in formats) and "barrier" not in impl
        )  # bfp2/bfp0 requires barrier synchronization
        if not ("bfp0" in formats and "constexpr" in impl)  # bfp0 does not work with constexpr
        if not (M == 1 and K == 7168 and N_per_core == 256 and formats == ["bfp8"])  # Not enough memory
        if not (M == 1 and K == 7168 and N_per_core == 256 and "constexpr" in impl)  # Not enough program memory
    ],
)
def test_matmul_custom_compressed_multicore(device, M, K, N_per_core, formats, num_cores, impl):
    """Multicore tests with automatic format assignment across all implementations."""
    if "runtime" in impl and num_cores == 32 and formats != ["bfp8"]:
        pytest.skip("FIXME: PCC ERROR")
    if (
        (M, K, N_per_core) == (1, 512, 256)
        and formats != ["bfp8"]
        and impl in ("constexpr_unroll barrier", "constexpr_compact barrier")
    ):
        pytest.skip("FIXME: PCC ERROR")
    _run_matmul_custom_compressed(device, M, K, N_per_core * num_cores, impl=impl, formats=formats, num_cores=num_cores)


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_matmul_custom_compressed_clustered(device, M, K, N, impl):
    """Top half bfp8, bottom half bfp4. Exactly 2 runs."""
    _run_matmul_custom_compressed_clustered(device, M, K, N, impl=impl)


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
@pytest.mark.parametrize("interleave_n", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_matmul_custom_compressed_interleaved(device, M, K, N, interleave_n, impl):
    """Alternating format assignment every N tiles."""
    _run_matmul_custom_compressed_interleaved_by_n(device, M, K, N, impl=impl, interleave_n=interleave_n)


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_matmul_custom_compressed_hybrid(device, M, K, N, impl):
    """Mixed interleaving: sections of by-2, by-4, by-8, by-16 in one tensor."""
    _run_matmul_custom_compressed_hybrid(device, M, K, N, impl=impl)


@pytest.mark.parametrize(
    "M,K,N,formats",
    [
        (M, K, N, formats)
        for M, K, N in SHAPES
        for formats in FORMATS
        if not (
            ((K // 32) * (N // 32)) <= 8 and len(formats) > 2
        )  # Small shapes with many formats may not have enough tiles
        if not (M == 1 and K == 7168 and N == 256 and formats == ["bfp8"])  # Not enough memory
    ],
)
def test_matmul_custom_compressed_single_core_optimized(device, M, K, N, formats):
    """Single core tests with automatic format assignment."""
    _run_matmul_custom_compressed(device, M, K, N, impl="new", formats=formats)


@pytest.mark.parametrize(
    "M,K,N_per_core,formats,num_cores",
    [
        (M, K, N_per_core, formats, num_cores)
        for M, K, N_per_core in SHAPES
        for formats in FORMATS
        for num_cores in [2, 13, 32]
        if not (M == 1 and K == 7168 and N_per_core == 256 and formats == ["bfp8"])  # Not enough memory
    ],
)
def test_matmul_custom_compressed_multicore_optimized(device, M, K, N_per_core, formats, num_cores):
    """Multicore tests with automatic format assignment."""
    if num_cores == 32 and formats != ["bfp8"]:
        pytest.skip("FIXME: PCC ERROR")
    _run_matmul_custom_compressed(
        device, M, K, N_per_core * num_cores, impl="new", formats=formats, num_cores=num_cores
    )


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
def test_matmul_custom_compressed_clustered_optimized(device, M, K, N):
    """Top half bfp8, bottom half bfp4. Exactly 2 runs."""
    _run_matmul_custom_compressed_clustered(device, M, K, N, impl="new")


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
@pytest.mark.parametrize("interleave_n", [2, 4, 8, 16, 32])
def test_matmul_custom_compressed_interleaved_optimized(device, M, K, N, interleave_n):
    """Alternating format assignment every N tiles."""
    _run_matmul_custom_compressed_interleaved_by_n(device, M, K, N, impl="new", interleave_n=interleave_n)


@pytest.mark.parametrize("M,K,N", [s for s in SHAPES if s != (1, 7168, 256)])
def test_matmul_custom_compressed_hybrid_optimized(device, M, K, N):
    """Mixed interleaving: sections of by-2, by-4, by-8, by-16 in one tensor."""
    _run_matmul_custom_compressed_hybrid(device, M, K, N, impl="new")
