# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

TORCH_TO_TTNN = {
    torch.float32: ttnn.float32,
    torch.bfloat16: ttnn.bfloat16,
    torch.uint32: ttnn.uint32,
    torch.int32: ttnn.int32,
    torch.uint16: ttnn.uint16,
    torch.uint8: ttnn.uint8,
}


@pytest.mark.parametrize("use_device", [True, False])
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", TORCH_BENCH_TYPES)
@pytest.mark.parametrize("ttnn_dtype", TTNN_BENCH_TYPES)
@pytest.mark.parametrize("size", [1024])
def test_benchmark_from_torch(benchmark, device, use_device, ttnn_dtype, torch_dtype, ttnn_layout, size):
    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pytest.skip("ROW_MAJOR_LAYOUT not supported with bfloat8_b/bfloat4_b")

    if ttnn_layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype == TORCH_TO_TTNN.get(torch_dtype, None):
        pytest.skip(
            "Don't benchmark tensors which are already in the correct layout and dtype.(borrowed data + data transfer)"
        )

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
            fast_approx=True,
        )

        if not use_device:
            ttnn.to_device(ttnn_tensor, device=device)

        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=2, warmup_rounds=1)
