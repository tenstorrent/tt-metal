# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does a decode step cost what its key width says, or what its **tile count** says?

The in-place KV cache needs a buffer 32 rows wider than the window it serves, and
measured on Blackhole that widening cost +1.21 ms on a 6.74 ms step -- 18 % more time
for 8 % more data. Something other than volume is setting that price.

The suspect is divisibility. 384 rows is 12 tiles; 416 is 13, which is prime. Every
op on the key axis splits its tiles across cores, and a tile count that divides badly
leaves one core holding an extra tile while the rest idle -- so the step pays for
`ceil(tiles / cores)`, not for `tiles`. If that is what is happening, then 512 rows
(16 tiles, highly composite) could be *cheaper* than 416, and the in-place design's
+0.82 ms mechanism would go from buried to banked.

If instead cost rises monotonically with width, the widening is simply volume, the
in-place design cannot pay for itself at any scratch size, and that is the end of it.

Either answer is worth having, and it is one sweep.

    python models/demos/cosyvoice/scripts/probe_key_width.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

WIDTHS = (320, 352, 384, 416, 448, 480, 512)
PREFIX = 209
REPS = 32


def main() -> int:
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path

    path = default_weights_path().replace("hift_", "llm_")
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        bag = WeightBag.load(path)
        meta = bag.meta["ar_decoder"]
        dec = TtARDecoder(device, bag.sub("llm"), meta)
        torch.manual_seed(0)
        prefix = torch.randn(1, PREFIX, meta["input_size"]) * 0.1
        x = torch.randn(1, 1, meta["input_size"]) * 0.1

        def dev(v):
            return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        print(f"\n  {'key width':>10}{'tiles':>7}{'ms/step':>10}{'tok/s':>9}{'us per tile':>13}")
        print("  " + "-" * 49)
        base = None
        for w in WIDTHS:
            ys, caches = dec.forward_chunk_fixed(
                dev(prefix),
                dec.empty_cache(w, PREFIX),
                w,
                valid=PREFIX,
                mask=dev(right_aligned_bias(w, PREFIX, PREFIX, causal=True)),
            )
            ttnn.deallocate(ys)
            step = TracedDecodeStep(dec, w).capture()
            step.seed(caches)
            TtARDecoder.free_caches(caches)
            for i in range(2):
                step.step(x, PREFIX + 1 + i)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for i in range(REPS):
                step.step(x, PREFIX + 3 + i)
                ttnn.synchronize_device(device)
            ms = (time.perf_counter() - t0) / REPS * 1e3
            step.release()
            base = ms if base is None else base
            tiles = w // 32
            print(f"  {w:>10}{tiles:>7}{ms:>10.2f}{1e3/ms:>9.1f}{ms * 1e3 / tiles:>13.1f}")

        print("\n  Monotonic in width  -> the widening is volume; in-place cannot pay for itself.")
        print("  Non-monotonic       -> pick a buffer width whose tile count divides well.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
