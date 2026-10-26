# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-3 / R2-4 micro-benchmark: head split, per-head q/k RMSNorm and RoPE of one time layer at the paper shape.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/micro_qk_heads.py \
        --out models/experimental/t0_forecast/doc/optimize/iter_02/r2_3_micro.json

From the flat qkv (R, 1, S_pad, 3d) to q, k, v (R, H, S_pad, hd), q and k normed per head and rotated:
    current   nlp_create_qkv_heads, rms_norm q and k (gains), rotary_embedding_llama q and k
    folded    as current, the gains folded into per-q / per-k RoPE tables (g cos, swap(g) sin), norms without gain
    joint     nlp_create_qkv_heads as 2H query heads + H/2 "kv" heads (q|k, v halves), one norm, one RoPE with
              per-head tables, slices q / k and concat v
    wan       nlp_create_qkv_heads, then per q / k wan_fused_rmsnorm pre + post (norm and RoPE in one pass) on the
              (1, 1, R H S_pad, hd) view, folded tables tiled over R H
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

R, H, S, D = 1024, 8, 96, 128
DM = H * D


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def time_op(device, fn, batches=5, calls=5):
    import ttnn

    def free(outs):
        for o in outs:
            ttnn.deallocate(o)

    for _ in range(2):
        free(fn())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        outs = [fn() for _ in range(calls)]
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) / calls)
        for o in outs:
            free(o)
    return statistics.median(samples)


def main():
    import ttnn

    from models.experimental.t0_forecast.tt.common import TtT0Precision, rms_norm, upload_norm
    from models.experimental.t0_forecast.tt.host import pair_rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma list; merge into an existing --out")
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(7)
    qkv = torch.randn(R, S, 3 * DM, generator=gen)
    gq, gk = (1 + 0.2 * torch.randn(D, generator=gen) for _ in range(2))
    pos = torch.arange(S, dtype=torch.float32)[:, None]
    theta = pos * (10000 ** (-torch.arange(0, D, 2) / D)).repeat_interleave(2)[None]
    xpos = (1.02 ** (pos / 32)) * torch.ones(1, D)
    cq, sq, ck, sk = torch.cos(theta) * xpos, torch.sin(theta) * xpos, torch.cos(theta) / xpos, torch.sin(theta) / xpos
    rot = pair_rotation(D)
    swap = lambda g: g.reshape(-1, 2).flip(-1).reshape(-1)

    t = qkv.reshape(R, S, 3, H, D).permute(2, 0, 3, 1, 4)  # (3, R, H, S, D)
    norm = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)
    rope = lambda x, c, s: x * c + (x @ rot) * s
    ref = (rope(norm(t[0]) * gq, cq, sq), rope(norm(t[1]) * gk, ck, sk), t[2])

    device = ttnn.open_device(device_id=0)
    p = TtT0Precision.accuracy()
    exact = p.exact_kernel_config()
    up = lambda x, dtype=ttnn.bfloat16: ttnn.from_torch(
        x.to(torch.bfloat16) if dtype == ttnn.bfloat16 else x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    tab = lambda x: up(x.reshape(1, 1, S, D))
    trans = up(pair_rotation(32).reshape(1, 1, 32, 32))
    qkv_dev = up(qkv.reshape(R, 1, S, 3 * DM))
    result = {"shape": {"R": R, "H": H, "S_pad": S, "hd": D}}

    def ttrope(x, c, s):
        out = ttnn.experimental.rotary_embedding_llama(
            x, c, s, trans, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=exact
        )
        ttnn.deallocate(x)
        return out

    def heads(n=H, kv=None):
        return ttnn.experimental.nlp_create_qkv_heads(
            qkv_dev, num_heads=n, num_kv_heads=kv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def normed(x, gain):
        out = rms_norm(x, gain, 1e-8, p)
        ttnn.deallocate(x)
        return out

    g_q, g_k = upload_norm(device, gq), upload_norm(device, gk)
    tables = [tab(x) for x in (cq, sq, ck, sk)]

    def current():
        q, k, v = heads()
        q, k = normed(q, g_q), normed(k, g_k)
        return ttrope(q, tables[0], tables[1]), ttrope(k, tables[2], tables[3]), v

    folded_tabs = [tab(cq * gq), tab(sq * swap(gq)), tab(ck * gk), tab(sk * swap(gk))]

    def folded():
        q, k, v = heads()
        q, k = normed(q, None), normed(k, None)
        return ttrope(q, folded_tabs[0], folded_tabs[1]), ttrope(k, folded_tabs[2], folded_tabs[3]), v

    jc = torch.cat([(cq * gq).expand(H, S, D), (ck * gk).expand(H, S, D)]).reshape(1, 2 * H, S, D)
    js = torch.cat([(sq * swap(gq)).expand(H, S, D), (sk * swap(gk)).expand(H, S, D)]).reshape(1, 2 * H, S, D)
    joint_tabs = [up(jc), up(js)]

    def joint():
        qk, va, vb = heads(2 * H, H // 2)
        v = ttnn.concat([va, vb], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(va)
        ttnn.deallocate(vb)
        qk = ttrope(normed(qk, None), joint_tabs[0], joint_tabs[1])
        q = ttnn.slice(qk, (0, 0, 0, 0), (R, H, S, D), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(qk, (0, H, 0, 0), (R, 2 * H, S, D), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qk)
        return q, k, v

    tiled = lambda x: up(x.expand(R * H, S, D).reshape(1, 1, R * H * S, D))
    wan_tabs = [tiled(cq * gq), tiled(sq * swap(gq)), tiled(ck * gk), tiled(sk * swap(gk))]
    wan_g = [None, None]  # gains folded into the tables

    def wan_one(x, gain, c, s):
        flat = ttnn.reshape(x, (1, 1, R * H * S, D))
        stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(flat, compute_kernel_config=exact, dtype=ttnn.float32)
        out = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
            flat,
            stats,
            epsilon=1e-8,
            num_heads_per_device=1,
            weight=gain,
            transformation_mat=trans,
            rope_cos=c,
            rope_sin=s,
            compute_kernel_config=exact,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(stats)
        ttnn.deallocate(x)
        return ttnn.reshape(out, (R, H, S, D))

    def wan():
        q, k, v = heads()
        return wan_one(q, wan_g[0], wan_tabs[0], wan_tabs[1]), wan_one(k, wan_g[1], wan_tabs[2], wan_tabs[3]), v

    try:
        runs = (("current", current), ("folded", folded), ("joint", joint), ("wan", wan))
        if args.only:
            result.update(json.loads(args.out.read_text()) if args.out.is_file() else {})
            runs = [r for r in runs if r[0] in args.only.split(",")]
        for name, fn in runs:
            try:
                outs = fn()
                got = [ttnn.to_torch(o).float() for o in outs]
                for o in outs:
                    ttnn.deallocate(o)
                result[name] = {f"pcc_{n}": pcc(r, g) for n, r, g in zip("qkv", ref, got)}
                result[name]["ms"] = time_op(device, fn) * 1e3
            except Exception as e:  # noqa: BLE001
                result[name] = {"error": str(e)[:400]}
            print(name, result[name], flush=True)
    finally:
        ttnn.close_device(device)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
