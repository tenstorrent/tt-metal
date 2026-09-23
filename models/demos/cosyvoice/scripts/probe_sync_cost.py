# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Would 2-CQ double-buffering pay here? Measures the idle `synchronize_device` cost.

A second command queue could hide the embedding row's host-to-device transfer in the
per-token tail (PERF.md Part II §1.6). That only pays if a `synchronize_device` on an
idle queue costs less than the transfer it hides, so this measures that cost on the
local board before any pipeline is built.

    python models/demos/cosyvoice/scripts/probe_sync_cost.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

REPS = 200
EMBED_MS = 0.092  # measured: embedding row -> device, per token (see PERF.md)


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, num_command_queues=2)
    try:
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.synchronize_device(device)
        idle_ms = (time.perf_counter() - t0) * 1e3 / REPS

        row = torch.randn(1, 1, 1024)
        t0 = time.perf_counter()
        for _ in range(REPS):
            t = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.deallocate(t)
        ttnn.synchronize_device(device)
        h2d_ms = (time.perf_counter() - t0) * 1e3 / REPS

        print(f"\n  mean of {REPS}, Blackhole p150a, num_command_queues=2")
        print(f"    synchronize_device on an idle queue   {idle_ms:.4f} ms")
        print(f"    embedding row host -> device          {h2d_ms:.4f} ms")
        print(f"    (PERF.md's per-token figure           {EMBED_MS:.3f} ms)")
        verdict = "PAYS" if idle_ms < h2d_ms else "LOSES"
        print(f"\n    a second queue would {verdict}: it can hide at most {h2d_ms:.4f} ms")
        print(f"    and costs {idle_ms:.4f} ms per token to synchronise.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
