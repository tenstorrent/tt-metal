# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def test_benchmark_from_torch_with_device_no_conversion(benchmark):
    """Benchmark for from_torch with device specified (without format conversion)"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    def from_torch_with_device():
        ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    benchmark.pedantic(from_torch_with_device, iterations=100, rounds=10, warmup_rounds=1)
    ttnn.close_device(device)


def test_benchmark_from_torch_with_device_with_conversion(benchmark):
    """Benchmark for from_torch with device specified (with format conversion)"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    def from_torch_with_device_conversion():
        ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    benchmark.pedantic(from_torch_with_device_conversion, iterations=10, rounds=5, warmup_rounds=1)
    ttnn.close_device(device)


def test_benchmark_copy_host_to_device_tensor(benchmark):
    """Benchmark for copy_host_to_device_tensor"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    # Create host tensor
    host_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Create device tensor
    device_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1024, 1024]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    def copy_host_to_device():
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    benchmark.pedantic(copy_host_to_device, iterations=100, rounds=10, warmup_rounds=1)
    device_tensor.deallocate(force=True)
    ttnn.close_device(device)


def test_benchmark_copy_device_to_host_tensor(benchmark):
    """Benchmark for copy_device_to_host_tensor"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    # Create device tensor with data
    device_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create host tensor for copy
    host_tensor = ttnn.allocate_tensor_on_host(ttnn.Shape([1024, 1024]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)

    def copy_device_to_host():
        ttnn.copy_device_to_host_tensor(device_tensor, host_tensor, blocking=True)

    benchmark.pedantic(copy_device_to_host, iterations=100, rounds=10, warmup_rounds=1)
    device_tensor.deallocate(force=True)
    ttnn.close_device(device)


def test_benchmark_from_device(benchmark):
    """Benchmark for from_device"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    # Create device tensor
    device_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def from_device():
        ttnn.from_device(device_tensor, blocking=True)

    benchmark.pedantic(from_device, iterations=100, rounds=10, warmup_rounds=1)
    device_tensor.deallocate(force=True)
    ttnn.close_device(device)


def test_benchmark_to_torch_from_device(benchmark):
    """Benchmark for to_torch (from device)"""
    torch_tensor = torch.rand((1024, 1024), dtype=torch.bfloat16)
    device = ttnn.open_mesh_device(0)

    # Create device tensor
    device_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def to_torch_from_device():
        device_tensor.to_torch()

    benchmark.pedantic(to_torch_from_device, iterations=100, rounds=10, warmup_rounds=1)
    device_tensor.deallocate(force=True)
    ttnn.close_device(device)
