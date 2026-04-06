# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def _get_asignment_for_distribution(distribution, K, N, formats):
    FORMAT_MAP = {
        "bfp8": 0,
        "bfp4": 1,
        "bfp2": 2,
        "bfp0": 3,
    }
    formats = [FORMAT_MAP[fmt] for fmt in formats]

    num_tiles = (K // 32) * (N // 32)
    if distribution == "clustered":
        num_runs = min(len(formats), num_tiles)
        run_lenth = num_tiles // num_runs
        last_run_length = num_tiles - run_lenth * (num_runs - 1)
        assignment = []
        for i in range(num_runs - 1):
            assignment.extend([formats[i]] * run_lenth)
        assignment.extend([formats[num_runs - 1]] * last_run_length)
        assignment = np.array(assignment, dtype=np.int8).reshape((K // 32, N // 32))
    elif distribution.startswith("interleaved "):
        interleave_n = int(distribution.split(" ")[1])
        num_blocks = (num_tiles + interleave_n - 1) // interleave_n
        assignment = []
        for i in range(num_blocks - 1):
            fmt = formats[i % len(formats)]
            assignment.extend([fmt] * interleave_n)
        fmt = formats[(num_blocks - 1) % len(formats)]
        assignment.extend([fmt] * (num_tiles - len(assignment)))
        assignment = np.array(assignment, dtype=np.int8).reshape((K // 32, N // 32))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return assignment


def _run_matmul_custom_compressed_benchmark(
    device,
    M,
    K,
    N,
    formats,
    implementation,
    distribution,
    pcc_threshold=0.98,
    threshold=0.993,
    tile_scaler=None,
):
    assert M == 1 or M == 8
    assert K % 32 == 0 and N % 32 == 0

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()
    if tile_scaler is not None:
        tile_scaler(torch_b, formats)
    else:
        scale_tiles_for_mixed_formats(torch_b, formats)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    a_tile = ttnn.Tile([M, 32])
    out_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, N], ttnn.ShardOrientation.ROW_MAJOR)
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    if distribution == "random":
        bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
        assigner = CompressedTensorAssigner(
            metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae
        )
        ct = CompressedTensor.from_torch(torch_b, assigner, device=device, memory_config=b_mem_config)
    else:
        assignment = _get_asignment_for_distribution(distribution, K, N, formats)
        ct = CompressedTensor(torch_b, assignment, device=device, memory_config=b_mem_config)

    logger.info(f"Custom compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")
    assignment = ct.get_assignment()
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
    logger.info(f"Assignment: {len(flat)} tiles, {len(runs)} runs, first 10 runs: {runs[:10]}")

    num_tiles = (K // 32) * (N // 32)
    is_interleaved = distribution.startswith("interleaved")
    is_random = distribution == "random"
    interleave_n = int(distribution.split(" ")[1]) if is_interleaved else None

    all_formats_present = num_tiles >= len(formats)
    if is_random:
        all_formats_present = num_tiles > 8
    if is_interleaved:
        num_blocks = (num_tiles + interleave_n - 1) // interleave_n
        all_formats_present = num_blocks >= len(formats)

    if all_formats_present:
        counts = ct.tile_counts
        for fmt in formats:
            assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"
    else:
        logger.warning(
            f"Number of tiles ({num_tiles}) is less than number of formats ({len(formats)}), cannot verify all formats are present in the assignment"
        )

    torch_expected = (torch_a.float() @ torch_b).bfloat16()

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output, impl=implementation)

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)
    assert passing, pcc_message


SHAPES = [
    (1, 64, 32),
    (1, 64, 64),
    # (1,  128, 512),
    (1, 256, 32),
    (1, 256, 128),
    (1, 512, 128),
    (1, 512, 256),
    (1, 1536, 128),
    (1, 2048, 32),
    (1, 3584, 32),
    (1, 7168, 32),
    (1, 7168, 64),
    # (1, 7168, 160),
    # (1, 7168, 256),
    (1, 8192, 64),
    # (8,  256, 512),
    # (8,  512, 512),
    (8, 576, 256),
    # (8,  576, 512),
]

SINGLE_FORMATS = [
    ["bfp8"],
    ["bfp4"],
    ["bfp2"],
    ["bfp2", "bfp0"],
]

MULTI_FORMATS = [
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

DISTRIBUTIONS = [
    "random",
    "clustered",
    "interleaved 2",
    "interleaved 4",
    "interleaved 8",
    "interleaved 16",
]


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", SINGLE_FORMATS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_matmul_custom_compressed_benchmark_baseline_single_format(device, M, K, N, formats, implementation):
    _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, implementation, "clustered")


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", MULTI_FORMATS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_matmul_custom_compressed_benchmark_baseline_mixed_formats(
    device, M, K, N, formats, implementation, distribution
):
    if ("bfp2" in formats or "bfp0" in formats) and "barrier" not in implementation:
        pytest.skip("bfp2 or bfp0 format requires barrier synchronization in the current implementation")
    _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, implementation, distribution)


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", SINGLE_FORMATS)
def test_matmul_custom_compressed_benchmark_optimized_single_format(device, M, K, N, formats):
    _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, "new", "clustered")


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", MULTI_FORMATS)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_matmul_custom_compressed_benchmark_optimized_mixed_formats(device, M, K, N, formats, distribution):
    _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, "new", distribution)


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", SINGLE_FORMATS)
def test_matmul_custom_compressed_benchmark_baseline_single_format_aggregated(device, M, K, N, formats):
    for implementation in IMPLEMENTATIONS:
        _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, implementation, "clustered", pcc_threshold=-2)


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", MULTI_FORMATS)
def test_matmul_custom_compressed_benchmark_baseline_mixed_formats_aggregated(device, M, K, N, formats):
    for implementation in IMPLEMENTATIONS:
        for distribution in DISTRIBUTIONS:
            if ("bfp2" in formats or "bfp0" in formats) and "barrier" not in implementation:
                continue
            _run_matmul_custom_compressed_benchmark(
                device, M, K, N, formats, implementation, distribution, pcc_threshold=-2
            )


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", SINGLE_FORMATS)
def test_matmul_custom_compressed_benchmark_optimized_single_format_aggregated(device, M, K, N, formats):
    _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, "new", "clustered", pcc_threshold=-2)


@pytest.mark.parametrize("M, K, N", SHAPES)
@pytest.mark.parametrize("formats", MULTI_FORMATS)
def test_matmul_custom_compressed_benchmark_optimized_mixed_formats_aggregated(device, M, K, N, formats):
    for distribution in DISTRIBUTIONS:
        _run_matmul_custom_compressed_benchmark(device, M, K, N, formats, "new", distribution, pcc_threshold=-2)
