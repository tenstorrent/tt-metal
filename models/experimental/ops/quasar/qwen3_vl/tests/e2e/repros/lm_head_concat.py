# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the LM head's concat of its vocab splits ([1,1,32,1280] x N along dim 3, bf16 DRAM), timed for growing N.

usage: lm_head_concat.py N...
"""
import sys
import time

import torch
import ttnn


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    for n in [int(a) for a in sys.argv[1:]] or [2, 8, 32]:
        parts = [torch.randn(1, 1, 32, 1280) for _ in range(n)]
        tts = [ttnn.from_torch(p.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for p in parts]
        t0 = time.time()
        kw = {"dim": 3, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        out = ttnn.concat(tts, **kw)
        got = ttnn.to_torch(out).float()
        ok = torch.equal(got, torch.cat([p.bfloat16().float() for p in parts], dim=3))
        print(
            f"concat n={n:3d} width={1280 * n:6d} exact={ok} {time.time() - t0:.1f}s",
            flush=True,
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
