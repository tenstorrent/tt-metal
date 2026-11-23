# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-time bench for wq_b_linear (run with scripts/run_safe_pytest.sh --profile).

Dispatches the op 1 warmup + 5 timed times on the single target shape; read the
DEVICE KERNEL DURATION column of the GenericOpDeviceOperation rows (take the median).
"""
import torch
import ttnn

from ttnn.operations.wq_b_linear import wq_b_linear


def test_bench(device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(6):
        y = wq_b_linear(tx, tw)
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
