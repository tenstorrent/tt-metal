# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Scratch check: concat with several blocks per core and uneven per-input widths (exercises the reader's wrap logic)."""
import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    cases = [
        ([(1, 1, 96, 64), (1, 1, 96, 96), (1, 1, 96, 32)], 3),
        ([(1, 2, 64, 32 * (i % 3 + 1)) for i in range(5)], 3),
        ([(1, 1, 32 * (i % 2 + 1), 64) for i in range(7)], 2),
        ([(2, 3, 32, 64) for _ in range(40)], 3),
        ([(1, 1, 160, 32 * (i % 4 + 1)) for i in range(35)], 3),
    ]
    for shapes, dim in cases:
        parts = [torch.randn(s) for s in shapes]
        tts = [ttnn.from_torch(p.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for p in parts]
        out = ttnn.concat(tts, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ok = torch.equal(ttnn.to_torch(out).float(), torch.cat([p.bfloat16().float() for p in parts], dim=dim))
        print(f"case n={len(shapes)} first={shapes[0]} dim={dim} exact={ok}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
