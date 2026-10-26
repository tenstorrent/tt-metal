# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-4 micro-benchmark: the fused post_add_norm op vs the 3-op chain (rms_norm(o) * gain, add, rms_norm) at
M = 65536 (paper) and 2048 (an odd row split), W = 1280, real-scale inputs. Prints ms/call, PCC and max |diff| of
h and xn against the chain (bar: <= 1.8 ms at 65536, PCC >= 0.99999).

    ttq run --model timesfm3 --part 0 --kind bench -- python models/experimental/timesfm3_forecast/tests/perf/bench_post_add_norm.py OUT.json
"""

import json
import sys
import time

import torch
import ttnn

from models.experimental.timesfm3_forecast.tt.common import TtPrecision, upload
from models.experimental.timesfm3_forecast.tt.model import EPS
from models.experimental.timesfm3_forecast.tt.ops.post_add_norm.op import post_add_norm

W, ITERS = 1280, 20


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def main(out=None):
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        cfg = TtPrecision.accuracy().norm_config()
        gen = torch.Generator().manual_seed(3)
        gain = upload(dev, (1 + 0.3 * torch.randn(1, W, generator=gen)))
        for m in (65536, 2048, 32 * 94):
            o = upload(dev, torch.randn(1, 1, m, W, generator=gen) * 3)
            r = upload(dev, torch.randn(1, 1, m, W, generator=gen) * 20)

            def chain():
                y = ttnn.rms_norm(
                    o, epsilon=EPS, weight=gain, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg
                )
                h = ttnn.add(r, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y)
                return h, ttnn.rms_norm(
                    h, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg
                )

            def fused():
                return post_add_norm(o, r, gain, epsilon=EPS)

            outs = {}
            for name, fn in (("chain", chain), ("fused", fused)):
                for _ in range(3):
                    for t in fn():
                        ttnn.deallocate(t)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    for t in fn():
                        ttnn.deallocate(t)
                ttnn.synchronize_device(dev)
                ms = (time.perf_counter() - t0) / ITERS * 1e3
                outs[name] = [ttnn.to_torch(t).float() for t in fn()]
                res[f"{name}@{m}"] = {"ms_per_call": ms}
            for i, k in enumerate(("h", "xn")):
                a, b = outs["chain"][i], outs["fused"][i]
                res[f"fused@{m}"][f"pcc_{k}"] = pcc(a, b)
                res[f"fused@{m}"][f"max_diff_{k}"] = float((a - b).abs().max())
            print(
                f"[POST_ADD_NORM] M={m}: chain {res[f'chain@{m}']['ms_per_call']:.3f} ms, {res[f'fused@{m}']}",
                flush=True,
            )
        if out:
            json.dump(res, open(out, "w"), indent=1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
