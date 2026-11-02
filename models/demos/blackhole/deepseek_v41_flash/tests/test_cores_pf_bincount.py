# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip masked_bincount (the routing prefix of the prefill MoE, 90 us in the layer): correctness against torch.bincount and device time (cores-for-prefill study)."""
import time

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cores_pf_bincount(device):
    dev = device
    N, K, E = 1024, 6, 384
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, E, (N, K), generator=g).to(torch.int32)
    ids[:5, :] = E  # a few sentinel rows (dropped)
    mask = torch.zeros(1, E + 1, dtype=torch.int32)
    mask[0, E] = -1
    t_ids = ttnn.to_layout(
        ttnn.from_torch(
            ids.to(torch.int32),
            device=dev,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.TILE_LAYOUT,
    )
    t_mask = ttnn.from_torch(
        mask, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.experimental.deepseek_prefill.masked_bincount(t_ids, t_mask, E, K)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out).reshape(-1)[:E].long()
    want = torch.bincount(ids[ids < E].flatten().long(), minlength=E)
    print(f"BINCOUNT correct {bool(torch.equal(got, want))}", flush=True)
    ttnn.deallocate(out)
    t = time.perf_counter()
    for _ in range(20):
        o = ttnn.experimental.deepseek_prefill.masked_bincount(t_ids, t_mask, E, K)
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    print(f"BINCOUNT {(time.perf_counter() - t) / 20 * 1e6:.1f} us per call", flush=True)
