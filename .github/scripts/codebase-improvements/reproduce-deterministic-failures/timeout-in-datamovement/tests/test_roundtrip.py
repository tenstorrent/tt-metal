# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test simple roundtrip without gather to isolate the bottleneck.
"""

import pytest
import torch
import ttnn
import time


def test_simple_roundtrip(device):
    """
    Move tensor to device and back without any operation.
    This tests raw transfer speed.
    """
    torch.manual_seed(42)

    # Same size as the gather test
    shape = [1, 151936]
    tensor = torch.randn(shape, dtype=torch.bfloat16)
    tensor_bytes = tensor.numel() * tensor.element_size()

    print(f"\n{'='*70}")
    print(f"SIMPLE ROUNDTRIP TEST (no gather)")
    print(f"Tensor: shape={shape}, size={tensor_bytes} bytes ({tensor_bytes/1024:.1f} KB)")
    print(f"{'='*70}")

    # To device
    t0 = time.perf_counter()
    ttnn_tensor = ttnn.from_torch(tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    t1 = time.perf_counter()

    # Back to host
    result = ttnn.to_torch(ttnn_tensor)
    t2 = time.perf_counter()

    to_device_ms = (t1 - t0) * 1000
    to_host_ms = (t2 - t1) * 1000

    print(f"\nTIMING:")
    print(f"  to_device:  {to_device_ms:8.2f} ms")
    print(f"  to_host:    {to_host_ms:8.2f} ms")

    if to_host_ms > 0:
        throughput = (tensor_bytes / (1024 * 1024)) / (to_host_ms / 1000)
        print(f"  to_host throughput: {throughput:.2f} MB/s")
    print(f"{'='*70}\n")

    assert result.shape == torch.Size(shape)


def test_larger_roundtrip(device):
    """
    Test with a larger tensor to get better throughput measurement.
    """
    torch.manual_seed(42)

    # Much larger tensor
    shape = [1, 1024 * 1024]  # 2 MB for bfloat16
    tensor = torch.randn(shape, dtype=torch.bfloat16)
    tensor_bytes = tensor.numel() * tensor.element_size()

    print(f"\n{'='*70}")
    print(f"LARGE ROUNDTRIP TEST (no gather)")
    print(f"Tensor: shape={shape}, size={tensor_bytes} bytes ({tensor_bytes/(1024*1024):.1f} MB)")
    print(f"{'='*70}")

    # To device
    t0 = time.perf_counter()
    ttnn_tensor = ttnn.from_torch(tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    t1 = time.perf_counter()

    # Back to host
    result = ttnn.to_torch(ttnn_tensor)
    t2 = time.perf_counter()

    to_device_ms = (t1 - t0) * 1000
    to_host_ms = (t2 - t1) * 1000

    print(f"\nTIMING:")
    print(f"  to_device:  {to_device_ms:8.2f} ms")
    print(f"  to_host:    {to_host_ms:8.2f} ms")

    if to_host_ms > 0:
        throughput = (tensor_bytes / (1024 * 1024)) / (to_host_ms / 1000)
        print(f"  to_host throughput: {throughput:.2f} MB/s")
    print(f"{'='*70}\n")

    assert result.shape == torch.Size(shape)


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", "--timeout=120"] + sys.argv[1:])
