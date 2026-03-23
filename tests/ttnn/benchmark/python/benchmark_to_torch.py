# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

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
    ttnn.uint8,
    ttnn.int32,
]


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [1024])
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
