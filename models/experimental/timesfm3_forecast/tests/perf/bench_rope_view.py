# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-6 micro-benchmark: ``rotary_embedding_llama`` on (R, 16, S, 96) vs a free leading-dim view (R * g, 16 / g, S, 96)
that keeps <= 8 rows per core. Paper shape after R1-1: R = 1024, S = 96. Prints ms per call and max abs diff.

    ttq run --model timesfm3 --part 0 --kind bench -- python models/experimental/timesfm3_forecast/tests/perf/bench_rope_view.py
"""

import json
import sys
import time

import torch
import ttnn

from models.experimental.timesfm3_forecast.common.decomposition import HEAD_DIM_PAD, pair_rotation, rope_tables
from models.experimental.timesfm3_forecast.tt.common import TILE, TtPrecision, upload

R, H, S, ITERS = 1024, 16, 96, 20


def main(out=None):
    dev = ttnn.open_device(device_id=0)
    try:
        cfg = TtPrecision.accuracy().norm_config()
        cos, sin = rope_tables(S, torch.float32)
        cos, sin = (upload(dev, t.reshape(1, 1, S, HEAD_DIM_PAD)) for t in (cos, sin))
        trans = upload(dev, pair_rotation(TILE).reshape(1, 1, TILE, TILE))
        x = upload(dev, torch.randn(R, H, S, HEAD_DIM_PAD))

        def rope(t):
            return ttnn.experimental.rotary_embedding_llama(
                t, cos, sin, trans, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg
            )

        def viewed(g):
            def f(t):
                y = rope(ttnn.reshape(t, (R * g, H // g, S, HEAD_DIM_PAD)))
                return ttnn.reshape(y, (R, H, S, HEAD_DIM_PAD))

            return f

        res, ref = {}, None
        for name, fn in [("current", rope), ("view g=2", viewed(2)), ("view g=4", viewed(4)), ("view g=8", viewed(8)), ("view g=16", viewed(16))]:
            for _ in range(3):
                ttnn.deallocate(fn(x))
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.deallocate(fn(x))
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) / ITERS * 1e3
            y = ttnn.to_torch(fn(x)).float()
            ref = y if ref is None else ref
            res[name] = {"ms_per_call": ms, "max_abs_diff_vs_current": float((y - ref).abs().max())}
            print(f"[ROPE] {name}: {ms:.3f} ms/call, max |diff| {res[name]['max_abs_diff_vs_current']:.3e}", flush=True)
        if out:
            json.dump(res, open(out, "w"), indent=1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
