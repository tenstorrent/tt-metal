# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def test_benchmark_from_torch_zero_copy(benchmark):
    # Zero copy from_torch: row_major + physical_shape == logical_shape2d + bfloat16->bfloat16
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=100, rounds=10, warmup_rounds=1)


def test_benchmark_from_torch_one_copy(benchmark):
    # One copy from_torch tile layout + physical_shape == logical_shape2d + bfloat16->bfloat16
    # - Copy on tilizing
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


def test_benchmark_from_torch_two_copy(benchmark):
    # Two copied from_torch tile layout + physical_shape != logical_shape2d + bfloat16->bfloat16
    # - First copy on padding to tile_size
    # - Second copy on tilizing
    torch_tensor = torch.rand((8096, 8100), dtype=torch.bfloat16)

    def from_torch():
        ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


TORCH_BENCH_TYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.uint8,
]

TTNN_BENCH_TYPES = [
    ttnn.float32,
    ttnn.bfloat16,
    ttnn.bfloat8_b,
    ttnn.bfloat4_b,
    ttnn.bfloat16,
    ttnn.uint8,
    ttnn.int32,
]


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [8192])
def test_benchmark_from_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, ttnn_layout, size, request):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")
    if torch_dtype in [torch.int32, torch.uint8, torch.int64]:
        torch_input_tensor = torch.randint(0, 100, (size, size), dtype=torch_dtype)
    else:
        torch_input_tensor = torch.rand((size, size), dtype=torch_dtype)

    def from_torch():
        ttnn_tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device if use_device else None,
            dtype=ttnn_dtype,
            layout=ttnn_layout,
        )

        if not use_device:
            moved = ttnn.to_device(ttnn_tensor, device=device)

        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [8192])
def test_benchmark_to_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, size, ttnn_layout):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")

    match ttnn_dtype:
        case ttnn.int32 | ttnn.uint8:
            tmp_torch = torch.randint(0, 100, (size, size), dtype=torch.int32)

        case _:
            tmp_torch = torch.rand(size, size, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(tmp_torch, device=device, dtype=ttnn_dtype, layout=ttnn_layout)

    def to_torch():
        ttnn.to_torch(ttnn_input_tensor, device=device if use_device else None, dtype=torch_dtype)
        ttnn.synchronize_device(device)

    benchmark.pedantic(to_torch, iterations=10, rounds=2, warmup_rounds=1)
