# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Profiling test to understand where time is spent in gather + to_torch
"""

import pytest
import torch
import ttnn
import numpy as np
import time

# The exact failing parameters from CI
INPUT_SHAPE = [1, 151936]
INDEX_SHAPE = [1, 151936]
DIM = -1


def test_gather_profiling(device):
    """
    Single iteration with detailed timing breakdown.
    """
    torch.manual_seed(42)
    torch_dtype = torch.bfloat16
    max_idx_val = INPUT_SHAPE[DIM]

    # Create input tensors
    input_tensor = torch.randn(INPUT_SHAPE, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, INDEX_SHAPE, dtype=torch.int64)

    # Log tensor sizes
    input_bytes = input_tensor.numel() * input_tensor.element_size()
    index_bytes = index.numel() * index.element_size()
    print(f"\n{'='*70}")
    print(
        f"INPUT TENSOR: shape={INPUT_SHAPE}, dtype={torch_dtype}, size={input_bytes} bytes ({input_bytes/1024:.1f} KB)"
    )
    print(f"INDEX TENSOR: shape={INDEX_SHAPE}, dtype=int64, size={index_bytes} bytes ({index_bytes/1024:.1f} KB)")
    print(f"{'='*70}")

    # Phase 1: Move input to device
    t0 = time.perf_counter()
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    t1 = time.perf_counter()

    # Phase 2: Move index to device
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)
    t2 = time.perf_counter()

    # Phase 3: Execute gather on device
    ttnn_gather = ttnn.gather(ttnn_input, DIM, index=ttnn_index)
    t3 = time.perf_counter()

    # Log output tensor info before reading back
    # Note: This doesn't trigger a read, just gets metadata
    print(f"OUTPUT TENSOR: shape={ttnn_gather.shape}")

    # Phase 4: Read result back - THIS IS WHERE TIMEOUT OCCURS
    result = ttnn.to_torch(ttnn_gather)
    t4 = time.perf_counter()

    # Calculate result size
    result_bytes = result.numel() * result.element_size()
    print(f"RESULT TENSOR: shape={result.shape}, size={result_bytes} bytes ({result_bytes/1024:.1f} KB)")

    # Timing breakdown
    input_to_device_ms = (t1 - t0) * 1000
    index_to_device_ms = (t2 - t1) * 1000
    gather_ms = (t3 - t2) * 1000
    to_torch_ms = (t4 - t3) * 1000
    total_ms = (t4 - t0) * 1000

    print(f"\n{'='*70}")
    print(f"TIMING BREAKDOWN:")
    print(f"  input to_device:  {input_to_device_ms:8.2f} ms")
    print(f"  index to_device:  {index_to_device_ms:8.2f} ms")
    print(f"  gather():         {gather_ms:8.2f} ms")
    print(f"  to_torch():       {to_torch_ms:8.2f} ms  <<< BOTTLENECK")
    print(f"  ---")
    print(f"  TOTAL:            {total_ms:8.2f} ms")
    print(f"{'='*70}")

    # Calculate throughput if to_torch is the bottleneck
    if to_torch_ms > 0:
        throughput_mb_s = (result_bytes / (1024 * 1024)) / (to_torch_ms / 1000)
        print(f"  to_torch throughput: {throughput_mb_s:.2f} MB/s")
    print(f"{'='*70}\n")

    # Write to file for easy retrieval
    with open("/tmp/gather_profiling.log", "w") as f:
        f.write(f"input_to_device={input_to_device_ms:.2f}ms\n")
        f.write(f"index_to_device={index_to_device_ms:.2f}ms\n")
        f.write(f"gather={gather_ms:.2f}ms\n")
        f.write(f"to_torch={to_torch_ms:.2f}ms\n")
        f.write(f"total={total_ms:.2f}ms\n")
        f.write(f"result_bytes={result_bytes}\n")

    assert result.shape == torch.Size(INDEX_SHAPE)


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", "--timeout=60"] + sys.argv[1:])
