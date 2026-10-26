# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R3-2 micro-benchmark: the fused rope_qk_norm op vs the chain it replaces (RoPE on the R2-1 view, then rms_norm
with the gain, for q and k), against an fp32 torch reference of the same math. Shapes (R, 16, S, 96): the paper
rows (R = 1024, S = 64) and an odd split (R = 37, S = 96); head lanes 80..95 are zero, as in the model. Bar:
<= 2.5 ms at the paper shape, PCC vs the chain >= 0.99999.

    ttq run --model timesfm3 --part 0 --kind bench -- python models/experimental/timesfm3_forecast/tests/perf/bench_rope_qk_norm.py OUT.json
"""

import json
import sys
import time

import torch
import ttnn

from models.experimental.timesfm3_forecast.common.decomposition import (
    HEAD_DIM,
    HEAD_DIM_PAD,
    pair_rotation,
    rope_tables,
)
from models.experimental.timesfm3_forecast.tt.common import TILE, TtPrecision, upload
from models.experimental.timesfm3_forecast.tt.model import QK_EPS, ROPE_VIEW_GROUPS
from models.experimental.timesfm3_forecast.tt.ops.rope_qk_norm.op import rope_qk_norm

H, ITERS = 16, 20


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def reference(x, cos, sin, gain):
    rot = torch.zeros_like(x)
    rot[..., 0::2], rot[..., 1::2] = -x[..., 1::2], x[..., 0::2]
    y = x * cos + rot * sin
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + QK_EPS) * gain


def main(out=None):
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        cfg = TtPrecision.accuracy().norm_config()
        gen = torch.Generator().manual_seed(4)
        trans = upload(dev, pair_rotation(TILE).reshape(1, 1, TILE, TILE))
        gains = []
        for _ in range(2):
            g = 1 + 0.3 * torch.randn(1, HEAD_DIM_PAD, generator=gen)
            g[:, HEAD_DIM:] = 0
            gains.append(g)
        gq, gk = (upload(dev, g) for g in gains)
        for r, s in ((1024, 64), (37, 96)):
            cos_t, sin_t = rope_tables(s, torch.float32)
            cos, sin = (upload(dev, t.reshape(1, 1, s, HEAD_DIM_PAD)) for t in (cos_t, sin_t))
            xs = []
            for scale in (3.0, 6.0):
                x = torch.randn(r, H, s, HEAD_DIM_PAD, generator=gen) * scale
                x[..., HEAD_DIM:] = 0
                xs.append(x)
            q, k = (upload(dev, x) for x in xs)
            g = ROPE_VIEW_GROUPS

            def rope(t):
                y = ttnn.experimental.rotary_embedding_llama(
                    ttnn.reshape(t, (r * g, H // g, s, HEAD_DIM_PAD)),
                    cos,
                    sin,
                    trans,
                    is_decode_mode=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=cfg,
                )
                return ttnn.reshape(y, (r, H, s, HEAD_DIM_PAD))

            def chain():
                outs = []
                for t, gain in ((q, gq), (k, gk)):
                    y = rope(t)
                    outs.append(
                        ttnn.rms_norm(
                            y,
                            epsilon=QK_EPS,
                            weight=gain,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=cfg,
                        )
                    )
                    ttnn.deallocate(y)
                return outs

            def fused():
                return list(rope_qk_norm(q, k, cos, sin, trans, gq, gk, epsilon=QK_EPS))

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
                res[f"{name}@{r}x{s}"] = {"ms_per_call": (time.perf_counter() - t0) / ITERS * 1e3}
                outs[name] = [ttnn.to_torch(t).float() for t in fn()]
            refs = [reference(x, cos_t, sin_t, gain) for x, gain in zip(xs, gains)]
            for name in ("chain", "fused"):
                rec = res[f"{name}@{r}x{s}"]
                for i, qk in enumerate("qk"):
                    rec[f"pcc_ref_{qk}"] = pcc(outs[name][i], refs[i])
                    rec[f"max_diff_ref_{qk}"] = float((outs[name][i] - refs[i]).abs().max())
                    if name == "fused":
                        rec[f"pcc_chain_{qk}"] = pcc(outs[name][i], outs["chain"][i])
                        rec[f"max_diff_chain_{qk}"] = float((outs[name][i] - outs["chain"][i]).abs().max())
                print(f"[ROPE_QK_NORM] {name} R={r} S={s}: {rec}", flush=True)
            for t in (q, k, cos, sin):
                ttnn.deallocate(t)
        if out:
            json.dump(res, open(out, "w"), indent=1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
