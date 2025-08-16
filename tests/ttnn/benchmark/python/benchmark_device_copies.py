# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _make_host_tensor(shape):
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _make_device_tensor_like(host_tensor, device):
    return ttnn.allocate_tensor_on_device(host_tensor.shape, host_tensor.dtype, host_tensor.layout, device)


"""
from_torch with device specified(both with format conversion and without)
"""


def test_benchmark_from_torch_to_device_row_major_no_format_conversion(benchmark):
    # Host -> device, row-major layout (no conversion), same dtype
    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def run():
        ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        ttnn.synchronize_device(device)

    benchmark(run)


def test_benchmark_from_torch_to_device_tile_layout_with_format_conversion(benchmark):
    # Host -> device, tile layout (conversion), same dtype
    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)

    def run():
        ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        ttnn.synchronize_device(device)

    benchmark(run)


"""
copy_host_to_device_tensor
"""


def test_benchmark_copy_host_to_device_tensor(benchmark):
    device = ttnn.open_device(device_id=0)
    host_tensor = _make_host_tensor((8096, 8096))
    device_tensor = _make_device_tensor_like(host_tensor, device)

    def run():
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


def test_benchmark_copy_device_to_host_tensor(benchmark):
    # Prepare a device tensor by creating on device directly
    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)
    device_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    host_tensor = ttnn.allocate_tensor_on_host(device_tensor.shape, device_tensor.dtype, device_tensor.layout, device)

    def run():
        ttnn.copy_device_to_host_tensor(device_tensor, host_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


"""
from_device
"""


def test_benchmark_from_device(benchmark):
    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)
    device_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)

    def run():
        _ = ttnn.from_device(device_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)


"""
to_torch (from device)
"""


def test_benchmark_to_torch_from_device(benchmark):
    device = ttnn.open_device(device_id=0)
    torch_tensor = torch.rand((8096, 8096), dtype=torch.bfloat16)
    device_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)

    def run():
        _ = ttnn.to_torch(device_tensor)
        ttnn.synchronize_device(device)

    benchmark(run)
