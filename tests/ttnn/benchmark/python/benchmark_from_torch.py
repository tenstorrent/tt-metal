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
            enable_bf4_opt=True,
        )

        if not use_device:
            ttnn.to_device(ttnn_tensor, device=device)

        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("use_device", [True])
@pytest.mark.parametrize("ttnn_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("size", [1024])
@pytest.mark.parametrize("enable_bf4_opt", [True, False])
def test_benchmark_from_torch_fast_approx(
    benchmark, device, use_device, ttnn_dtype, torch_dtype, ttnn_layout, size, enable_bf4_opt
):
    # performacne for borrowed data must be the same
    # for enable_bf4_opt=True and enable_bf4_opt=False
    torch_input_tensor = torch.rand((size, size), dtype=torch_dtype, device=device)

    def from_torch():
        ttnn_tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device if use_device else None,
            dtype=ttnn_dtype,
            layout=ttnn_layout,
            enable_bf4_opt=enable_bf4_opt,
        )

        if not use_device:
            ttnn.to_device(ttnn_tensor, device=device)

        ttnn.synchronize_device(device)

    benchmark.pedantic(from_torch, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize(
    "shape",
    [
        ([1, 256, 7168, 2048]),
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("enable_bf4_opt", [True, False])
def test_from_torch_deep_seek_interleaved_moe_weights_galaxy(benchmark, mesh_device, shape, ttnn_dtype, enable_bf4_opt):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("Test is only valid on T3K (8 devices)")

    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.float32)

    def from_torch():
        ttnn_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            enable_bf4_opt=enable_bf4_opt,
        )
        ttnn.synchronize_device(mesh_device)

    benchmark.pedantic(from_torch, iterations=1, rounds=1, warmup_rounds=1)
