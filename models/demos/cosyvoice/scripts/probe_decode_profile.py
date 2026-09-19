# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-op-class cost of one AR decode layer, at the shapes it actually uses.

`probe_op_floor.py` established a ~6.3 us floor for a trivial op and
`probe_matmul_config.py` priced the four linears at 201 us per layer. That leaves
~5.4 ms of an 8.25 ms step unaccounted for across ~280 non-linear ops, and "the KV
cache is the largest remaining block" was an inference from tensor sizes rather than
a measurement. This prices each class directly so the claim can be checked.

Shapes are the real ones for d_model 1024, 16 heads, d_k 64, max_len 256:

    KV maintenance   slice / concat / copy on [1, 16, 256, 64]   (0.5 MB each)
    scores           [1,16,1,64] x [1,16,64,256]
    positional       [1,16,1,64] x [1,16,64,511] and its slice
    context          [1,16,1,256] x [1,16,256,64]
    norms/residuals  [1, 1, 1024]

    python models/demos/cosyvoice/scripts/probe_decode_profile.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, MAX_LEN, D = 16, 64, 256, 1024
N_LAYERS = 14


def timed(device, fn, reps=40):
    """Trace `fn` and time the replay, so host dispatch is out of the number."""
    for _ in range(2):
        out = fn()
        if out is not None:
            ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    kept = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) * 1e6 / reps

    ttnn.release_trace(device, tid)
    if kept is not None:
        ttnn.deallocate(kept)
    return us


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=201326592)
    try:
        t = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731

        kbuf = t(torch.randn(1, H, MAX_LEN, DK))
        knew = t(torch.randn(1, H, 1, DK))
        q = t(torch.randn(1, H, 1, DK))
        attn = t(torch.randn(1, H, 1, MAX_LEN))
        pt = t(torch.randn(1, H, DK, 2 * MAX_LEN - 1))
        bd = t(torch.randn(1, H, 1, 2 * MAX_LEN - 1))
        row = t(torch.randn(1, 1, D))
        mask = t(torch.zeros(1, 1, 1, MAX_LEN))
        g = t(torch.randn(1, D))
        b = t(torch.randn(1, D))

        def kv_trim():
            return ttnn.slice(kbuf, [0, 0, 1, 0], [1, H, MAX_LEN, DK])

        def kv_concat():
            trimmed = ttnn.slice(kbuf, [0, 0, 1, 0], [1, H, MAX_LEN, DK])
            out = ttnn.concat([trimmed, knew], dim=2)
            ttnn.deallocate(trimmed)
            return out

        def kv_writeback():
            ttnn.copy(kbuf, kbuf)
            return None

        cases = [
            ("KV slice (trim)", kv_trim, 2),
            ("KV slice+concat", kv_concat, 2),
            ("KV copy (writeback)", kv_writeback, 2),
            ("scores  q@k^T", lambda: ttnn.matmul(q, kbuf, transpose_b=True), 1),
            ("positional  q@pt", lambda: ttnn.matmul(q, pt), 1),
            ("positional slice", lambda: ttnn.slice(bd, [0, 0, 0, 0], [1, H, 1, MAX_LEN]), 1),
            ("context  attn@v", lambda: ttnn.matmul(attn, kbuf), 1),
            ("scale_mask_softmax", lambda: ttnn.scale_mask_softmax(attn, 0.125, mask), 1),
            ("layer_norm [1,1,1024]", lambda: ttnn.layer_norm(row, weight=g, bias=b, epsilon=1e-5), 2),
            ("add [1,1,1024]", lambda: ttnn.add(row, row), 4),
            ("qkv split", lambda: None, 0),
        ]

        print(f"\n  one decode layer, priced by op class (traced, mean of 40)")
        print(f"  {'op class':<26}{'us':>9}{'per layer':>11}{'x14':>10}")
        total = 0.0
        for label, fn, per_layer in cases:
            if per_layer == 0:
                continue
            us = timed(device, fn)
            sub = us * per_layer
            total += sub
            print(f"  {label:<26}{us:>9.1f}{sub:>11.1f}{sub * N_LAYERS / 1e3:>9.2f}ms")

        print(f"\n  measured subtotal (non-linear): {total:.1f} us/layer = {total * N_LAYERS / 1e3:.2f} ms")
        print(f"  four linears (probe_matmul_config): 201.2 us/layer = {201.2 * N_LAYERS / 1e3:.2f} ms")
        print(f"  measured decode step: 8.25 ms")

        kv = sum(timed(device, fn) * n for label, fn, n in cases if label.startswith("KV"))
        print(
            f"\n  KV maintenance alone: {kv:.1f} us/layer = {kv * N_LAYERS / 1e3:.2f} ms "
            f"({100 * kv * N_LAYERS / 1e3 / 8.25:.0f}% of the step)"
        )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
