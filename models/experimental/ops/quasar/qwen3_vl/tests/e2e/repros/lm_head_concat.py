# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the LM head's concat of its vocab splits ([1,1,32,1280] x N along dim 3, bf16 DRAM), timed for growing N.

usage: lm_head_concat.py [--grouped] N...   (--grouped: through the harness workaround, at most 32 inputs per concat)
"""
import sys
import time

import torch
import ttnn
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.op_overrides import _concat_in_groups


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    grouped = "--grouped" in sys.argv
    for n in [int(a) for a in sys.argv[1:] if a != "--grouped"] or [2, 8, 32]:
        parts = [torch.randn(1, 1, 32, 1280) for _ in range(n)]
        tts = [ttnn.from_torch(p.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev) for p in parts]
        t0 = time.time()
        kw = {"dim": 3, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        out = _concat_in_groups(ttnn.concat, (tts,), kw) if grouped else ttnn.concat(tts, **kw)
        got = ttnn.to_torch(out).float()
        ok = torch.equal(got, torch.cat([p.bfloat16().float() for p in parts], dim=3))
        print(
            f"concat{' grouped' if grouped else ''} n={n:3d} width={1280 * n:6d} exact={ok} {time.time() - t0:.1f}s",
            flush=True,
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
