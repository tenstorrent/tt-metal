# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    # torch tensor is row major by default, so this maps to:
    # No translation, tileize
    "to_layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize("dtype", [(torch.bfloat16, ttnn.float32), (torch.float32, ttnn.bfloat16)])
@pytest.mark.parametrize("tensor_size", [(8096, 8096)])
def test_benchmark_from_torch_with_format_conversion(benchmark, to_layout, dtype, tensor_size):
    from_dtype, to_dtype = dtype

    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand(tensor_size, dtype=from_dtype)

    def run():
        ttnn.from_torch(torch_tensor, dtype=to_dtype, layout=to_layout, device=device)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize(
    # No translation, tileize
    "from_layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize("dtype", [(ttnn.bfloat16, torch.float32), (ttnn.float32, torch.bfloat16)])
@pytest.mark.parametrize("tensor_size", [(8096, 8096)])
def test_benchmark_to_torch(benchmark, from_layout, dtype, tensor_size):
    from_dtype, to_dtype = dtype

    device = ttnn.open_device(device_id=0)
    ttnn_tensor = ttnn.rand(tensor_size, dtype=from_dtype, layout=from_layout, device=device)

    def run():
        _ = ttnn.to_torch(ttnn_tensor, dtype=to_dtype, device=device)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("tensor_size", [(8096, 8096)])
def test_benchmark_from_device(benchmark, tensor_size):
    device = ttnn.open_device(device_id=0)
    ttnn_tensor = ttnn.rand(tensor_size, device=device)

    def run():
        _ = ttnn.from_device(ttnn_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("tensor_size", [(8096, 8096)])
def test_benchmark_copy_host_to_device_tensor(benchmark, tensor_size):
    device = ttnn.open_device(device_id=0)
    ttnn_tensor = ttnn.rand(tensor_size, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)

    def run():
        ttnn.copy_host_to_device_tensor(host_tensor, ttnn_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("tensor_size", [(8096, 8096)])
def test_benchmark_copy_device_to_host_tensor(benchmark, tensor_size):
    device = ttnn.open_device(device_id=0)
    device_tensor = ttnn.rand(tensor_size, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(device_tensor.spec, device)

    def run():
        ttnn.copy_device_to_host_tensor(device_tensor, host_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)
