# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""P2d cost breakdown (after tests/p2d_probe.py found its promotion no faster than option B's refill): per user, the
PP pass of the window (2,048 tokens), the drafter fill (_fill_context, one bucket), the device-side taps of the window
(_context_taps: slices, masks, reduce-scatter), the host round trip of those taps as device-format host tensors
(from_device / to_device, no conversion) vs as torch (tap_buffers), and a 256-token PP pass.
usage: python -m qwen36_resident.tests.p2d_parts [--prompt-len 6145] [--reps 3]
"""
import argparse
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows
from qwen36_resident.tests.drafter_refill import CORPUS
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=6145)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"][1000 : 1000 + a.prompt_len].tolist()
    mesh = open_mesh()
    try:
        eng = ResidentDFlashRows(
            mesh, Q.Checkpoint(Q.checkpoint_dir()), users=4, max_pos=a.prompt_len + 512, R=4, prefill=True
        )
        W, pp = eng.window, eng.pp
        n = a.prompt_len - 1
        s0 = max(0, n - W) // 64 * 64
        sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
        t = {}

        def lap(k, fn):
            sync()
            t0 = time.perf_counter()
            r = fn()
            sync()
            t.setdefault(k, []).append(time.perf_counter() - t0)
            return r

        for _ in range(a.reps):
            lap("pp pass 2048", lambda: (pp.reset(), pp.run(torch.as_tensor(ids[s0:n]))))
            lap("fill (one bucket)", lambda: eng._fill_context(n - s0, phys=0, offset=s0))
            dev = lap("context taps (device)", lambda: eng._context_taps(0, n - s0))
            host = lap("taps -> host (device format)", lambda: [ttnn.from_device(x) for x in dev])
            back = lap(
                "taps -> device (device format)",
                lambda: [ttnn.to_device(h, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for h in host],
            )
            lap("fill from uploaded taps", lambda: eng.drafter.fill_context(back, s0, phys=0, valid_len=n - s0))
            for x in dev + back:
                ttnn.deallocate(x)
            lap("pp pass 256", lambda: (pp.reset(), pp.run(torch.as_tensor(ids[n - 256 : n]))))
        for k, v in t.items():
            print(f"{k:32s} {1e3 * sorted(v)[len(v) // 2]:7.1f} ms (median of {len(v)})", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
