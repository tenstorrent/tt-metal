# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# This file contains a set of benchmark that test the transfer performance between
# host and device with and without format conversions.

# This is meant to test if any modifications of tt-nn and tt-metal will regress
# the performance of data transfer.

# Currently the benchmark measures performance in terms of time to operate on a
# fixed sized tensor. While benchmarking for throughput maybe more appropriate,
# this is currently not implemented due to the limitations of pytest.benchmark .

import torch
import ttnn
import pytest

DEFAULT_DTYPE = ttnn.bfloat16
DEFAULT_LAYOUT = ttnn.TILE_LAYOUT
DEFAULT_TENSOR_SIZE = (8096, 8096)


@pytest.mark.parametrize(
    # torch tensor is row major by default, so this maps to:
    # No translation, tileize
    "to_layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize("dtype", [(torch.bfloat16, ttnn.float32), (torch.float32, ttnn.bfloat16)])
@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_benchmark_from_torch_with_format_conversion(benchmark, to_layout, dtype, device_id):
    from_dtype, to_dtype = dtype

    device = ttnn.open_device(device_id=device_id)
    torch_tensor = torch.rand(DEFAULT_TENSOR_SIZE, dtype=from_dtype)

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
@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_benchmark_to_torch(benchmark, from_layout, dtype, device_id):
    from_dtype, to_dtype = dtype

    device = ttnn.open_device(device_id=device_id)
    ttnn_tensor = ttnn.rand(DEFAULT_TENSOR_SIZE, dtype=from_dtype, layout=from_layout, device=device)

    def run():
        _ = ttnn.to_torch(ttnn_tensor, dtype=to_dtype, device=device)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_benchmark_from_device(benchmark, device_id):
    device = ttnn.open_device(device_id=device_id)
    ttnn_tensor = ttnn.rand(DEFAULT_TENSOR_SIZE, dtype=DEFAULT_DTYPE, layout=DEFAULT_LAYOUT, device=device)

    def run():
        _ = ttnn.from_device(ttnn_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_benchmark_copy_host_to_device_tensor(benchmark, device_id):
    device = ttnn.open_device(device_id=device_id)
    ttnn_tensor = ttnn.rand(DEFAULT_TENSOR_SIZE, dtype=DEFAULT_DTYPE, layout=DEFAULT_LAYOUT, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(ttnn_tensor.spec, device)

    def run():
        ttnn.copy_host_to_device_tensor(host_tensor, ttnn_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_benchmark_copy_device_to_host_tensor(benchmark, device_id):
    device = ttnn.open_device(device_id=device_id)
    device_tensor = ttnn.rand(DEFAULT_TENSOR_SIZE, dtype=DEFAULT_DTYPE, layout=DEFAULT_LAYOUT, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(device_tensor.spec, device)

    def run():
        ttnn.copy_device_to_host_tensor(device_tensor, host_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)
