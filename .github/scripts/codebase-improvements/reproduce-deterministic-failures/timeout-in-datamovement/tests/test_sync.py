# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to understand if gather completes synchronously or if to_torch waits for it.
"""

import pytest
import torch
import ttnn
import time


def test_gather_with_explicit_sync(device):
    """
    Test with explicit synchronize after gather to see if gather is lazy.
    """
    torch.manual_seed(42)

    shape = [1, 151936]
    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    index = torch.randint(0, shape[-1], shape, dtype=torch.int64)

    print(f"\n{'='*70}")
    print(f"GATHER WITH EXPLICIT SYNC TEST")
    print(f"{'='*70}")

    # To device
    t0 = time.perf_counter()
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)
    t1 = time.perf_counter()

    # Execute gather
    ttnn_gather = ttnn.gather(ttnn_input, -1, index=ttnn_index)
    t2 = time.perf_counter()

    # Explicit sync to force gather to complete
    ttnn.synchronize_device(device)
    t3 = time.perf_counter()

    # Now read back (should be fast if sync already waited)
    result = ttnn.to_torch(ttnn_gather)
    t4 = time.perf_counter()

    to_device_ms = (t1 - t0) * 1000
    gather_launch_ms = (t2 - t1) * 1000
    sync_ms = (t3 - t2) * 1000
    to_torch_ms = (t4 - t3) * 1000

    print(f"\nTIMING BREAKDOWN:")
    print(f"  to_device:      {to_device_ms:8.2f} ms")
    print(f"  gather launch:  {gather_launch_ms:8.2f} ms  (just submitting command)")
    print(f"  sync():         {sync_ms:8.2f} ms  (waiting for gather to complete)")
    print(f"  to_torch():     {to_torch_ms:8.2f} ms  (just data transfer)")
    print(f"  ---")
    print(f"  TOTAL:          {(t4-t0)*1000:8.2f} ms")
    print(f"{'='*70}")

    result_bytes = result.numel() * result.element_size()
    if to_torch_ms > 0:
        throughput = (result_bytes / (1024 * 1024)) / (to_torch_ms / 1000)
        print(f"  to_torch throughput: {throughput:.2f} MB/s (after sync)")
    print(f"{'='*70}\n")

    assert result.shape == torch.Size(shape)


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", "--timeout=120"] + sys.argv[1:])
