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


import inspect


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.float16,
        torch.float32,
        torch.int32,
        torch.uint8,
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat16,
        ttnn.uint8,
        ttnn.int32,
    ],
)
def test_benchmark_from_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, ttnn_layout):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")
    smaller = 8
    print("start test")
    with ttnn.tracy_zone(
        f"test_benchmark_from_torch[ttnn_dtype={ttnn_dtype}-torch_dtype={torch_dtype}-ttnn_layout={ttnn_layout}-use_device={use_device}]"
    ):
        height = int(8096 / smaller)
        width = int(8100 / smaller)
        with ttnn.tracy_zone("startup tensor creation"):
            if torch_dtype in [torch.int32, torch.uint8]:
                torch_input_tensor = torch.randint(0, 100, (height, width), dtype=torch_dtype)
            else:
                torch_input_tensor = torch.rand((height, width), dtype=torch_dtype)

        def from_torch():
            with ttnn.tracy_zone("bench body"):
                ttnn_tensor = ttnn.from_torch(
                    torch_input_tensor,
                    device=device if use_device else None,
                    dtype=ttnn_dtype,
                    layout=ttnn_layout,
                )

                if not use_device:
                    with ttnn.tracy_zone("moving to device"):
                        moved = ttnn.to_device(ttnn_tensor, device=device)

        with ttnn.tracy_zone("benchmark run"):
            benchmark.pedantic(from_torch, iterations=10, rounds=5, warmup_rounds=1)


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.float16,
        torch.float32,
        torch.int32,
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.uint8,
        ttnn.int32,
    ],
)
def test_benchmark_to_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype):
    # if ttnn_dtype in [ttnn.bfloat8_b, ttnn.uint8]:
    #     pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")
    print("test starting")
    with ttnn.tracy_zone(
        f"test_benchmark_from_torch[ttnn_dtype={ttnn_dtype}-torch_dtype={torch_dtype}-use_device={use_device}]"
    ):
        height = int(8096 / 16)
        width = int(8100 / 16)
        with ttnn.tracy_zone("create initial tensor"):
            print("creating input tensor")
            match ttnn_dtype:
                case ttnn.int32:
                    tmp_torch = torch.randint(0, 100, (height, width), dtype=torch.int8)
                    ttnn_input_tensor = ttnn.from_torch(tmp_torch, device=device)

                case ttnn.uint8:
                    tmp_torch = torch.randint(0, 100, (height, width), dtype=torch.uint8)
                    ttnn_input_tensor = ttnn.from_torch(tmp_torch, device=device)

                case _:
                    ttnn_input_tensor = ttnn.rand((height, width), dtype=ttnn_dtype, device=device)

        def to_torch():
            with ttnn.tracy_zone("to_torch"):
                print("bench run")
                # ttnn.to_torch(ttnn_input_tensor, device=device if use_device else None, dtype=torch_dtype)

        with ttnn.tracy_zone("run benchmark"):
            benchmark.pedantic(to_torch, iterations=5, rounds=1, warmup_rounds=1)

    print("test done")
