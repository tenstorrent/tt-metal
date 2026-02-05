# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test how gather performance scales with tensor width.
"""

import pytest
import torch
import ttnn
import time


@pytest.mark.parametrize("width", [1024, 4096, 16384, 65536, 151936])
def test_gather_scaling(width, device):
    """
    Test gather performance for different tensor widths.
    """
    torch.manual_seed(42)

    shape = [1, width]
    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    index = torch.randint(0, width, shape, dtype=torch.int64)

    tensor_bytes = input_tensor.numel() * input_tensor.element_size()
    num_tiles = (width + 31) // 32  # Tiles in width dimension

    print(f"\n{'='*70}")
    print(f"WIDTH={width}, TILES={num_tiles}, SIZE={tensor_bytes/1024:.1f} KB")
    print(f"{'='*70}")

    # To device
    t0 = time.perf_counter()
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)
    t1 = time.perf_counter()

    # Gather
    ttnn_gather = ttnn.gather(ttnn_input, -1, index=ttnn_index)
    t2 = time.perf_counter()

    # Sync to isolate gather time
    ttnn.synchronize_device(device)
    t3 = time.perf_counter()

    # To torch
    result = ttnn.to_torch(ttnn_gather)
    t4 = time.perf_counter()

    to_device_ms = (t1 - t0) * 1000
    gather_launch_ms = (t2 - t1) * 1000
    gather_actual_ms = (t3 - t2) * 1000
    to_torch_ms = (t4 - t3) * 1000
    total_ms = (t4 - t0) * 1000

    # Calculate per-tile time
    per_tile_ms = gather_actual_ms / num_tiles if num_tiles > 0 else 0

    print(f"  to_device:     {to_device_ms:8.2f} ms")
    print(f"  gather launch: {gather_launch_ms:8.2f} ms")
    print(f"  gather actual: {gather_actual_ms:8.2f} ms ({per_tile_ms:.3f} ms/tile)")
    print(f"  to_torch:      {to_torch_ms:8.2f} ms")
    print(f"  TOTAL:         {total_ms:8.2f} ms")
    print(f"{'='*70}")

    assert result.shape == torch.Size(shape)


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s", "--timeout=300"] + sys.argv[1:])
