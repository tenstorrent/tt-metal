"""Is there a cheaper way to replicate-pad a few rows than concatenating the whole tensor?

`_replicate_pad_t` builds `[first]*pad_left + [x] + [last]*pad_right` and concats. At s6 that copies
165606 rows to add 6 -- and concat is 20 % of the decode (285 ms / 469 calls). Anything that writes
only the pad region, or that concats fewer pieces, is worth a fifth of the runtime.

Variants:
  today        12-piece concat, as shipped
  blocks       3-piece concat, edge rows materialised once via ttnn.repeat
  ttnn.pad     the dedicated pad op, if it supports this case
"""

import os
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _replicate_pad_t
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

SHAPES = [
    ("s4 C32  T41403", 32, 41403),
    ("s5 C16  T82806", 16, 82806),
    ("s6 C8   T165606", 8, 165606),
]
PAD_L, PAD_R = 5, 6
ITERS = int(os.environ.get("PAD_ITERS", "5"))


def timed(fn, device):
    fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(ITERS):
        s = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - s) * 1e3)
    return statistics.median(ts)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print(f"{'shape':<18} {'variant':<12} {'ms':>8}  {'exact vs today':>15}")
        print("-" * 60)
        for label, C, T in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            B = 2

            ref = ttnn.to_torch(_replicate_pad_t(xd, PAD_L, PAD_R, device)).float()
            print(
                f"{label:<18} {'today':<12} {timed(lambda: _replicate_pad_t(xd, PAD_L, PAD_R, device), device):>8.2f}"
                f"  {'(reference)':>15}"
            )

            def blocks():
                first = ttnn.slice(xd, [0, 0, 0], [B, 1, C])
                last = ttnn.slice(xd, [0, T - 1, 0], [B, T, C])
                lb = ttnn.repeat(first, ttnn.Shape([1, PAD_L, 1]))
                rb = ttnn.repeat(last, ttnn.Shape([1, PAD_R, 1]))
                return ttnn.concat([lb, xd, rb], dim=1)

            try:
                got = ttnn.to_torch(blocks()).float()
                ok = torch.equal(got, ref)
                print(f"{label:<18} {'blocks':<12} {timed(blocks, device):>8.2f}  {str(ok):>15}")
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<18} {'blocks':<12} {'-':>8}  FAIL {str(exc).splitlines()[0][:40]}")

            def padop():
                return ttnn.pad(xd, [(0, 0), (PAD_L, PAD_R), (0, 0)], value=0.0)

            try:
                got = ttnn.to_torch(padop()).float()
                # zero-pad, not replicate: compare only the interior to confirm the op works at all
                interior_ok = torch.equal(got[:, PAD_L : PAD_L + T, :], ref[:, PAD_L : PAD_L + T, :])
                print(f"{label:<18} {'ttnn.pad':<12} {timed(padop, device):>8.2f}  {'zeros/' + str(interior_ok):>15}")
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<18} {'ttnn.pad':<12} {'-':>8}  FAIL {str(exc).splitlines()[0][:40]}")
            print()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
