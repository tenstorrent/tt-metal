# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: ttnn.experimental.quasar.add for the vision QKV bias add, [1,1,R,3072] + [3072], timed per row count."""
import sys
import time

import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    b = torch.randn(3072)
    tb = ttnn.from_torch(b.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    for rows in [int(r) for r in (sys.argv[1:] or ["32", "64", "128", "256"])]:
        a = torch.randn(1, 1, rows, 3072)
        ta = ttnn.from_torch(a.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
        t0 = time.time()
        got = ttnn.to_torch(ttnn.experimental.quasar.add(ta, tb)).float()
        ref = a.bfloat16().float() + b.bfloat16().float()
        p = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
        print(f"rows={rows:4d} pcc={p:.5f} {time.time() - t0:.1f}s", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
