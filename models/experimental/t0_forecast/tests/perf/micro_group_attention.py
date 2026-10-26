# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-1 micro-benchmark gate: the group-attention core of one layer at the multivariate shape.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/micro_group_attention.py \
        --out models/experimental/t0_forecast/doc/optimize/iter_02/r2_1_micro.json

From the flat qkv (1, 1, R * S_pad, 3d) to the flat context (1, 1, R * S_pad, d), both paths with the q/k RMSNorm:
    current   permute to slot-major, (S_pad * B, G, 3d) blocks, heads, SDPA over G keys, permute back
    packed    (B, 1, G * S_pad, 3d) as it lies (rows of a series are contiguous), heads, SDPA over G * S_pad keys
              with an additive same-slot mask (1, 1, G * S_pad, G * S_pad)
Gate (A2): packed <= 40 ms, and its PCC against torch >= the current path's.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

B, G, S, H, D = 128, 8, 96, 8, 128
R, DM = B * G, H * D


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def time_op(device, fn, batches=5, calls=5):
    import ttnn

    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        outs = [fn() for _ in range(calls)]
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) / calls)
        for o in outs:
            ttnn.deallocate(o)
    return statistics.median(samples)


def reference(qkv):
    """(R * S, 3d) -> (R * S, d): per series and slot, attention over the G rows, q/k RMS-normed per head."""
    t = qkv.reshape(B, G, S, 3, H, D).permute(3, 0, 2, 4, 1, 5)  # (3, B, S, H, G, D)
    q, k, v = t[0], t[1], t[2]
    norm = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)
    ctx = torch.nn.functional.scaled_dot_product_attention(norm(q), norm(k), v)  # (B, S, H, G, D)
    return ctx.permute(0, 3, 1, 2, 4).reshape(R * S, DM)


def main():
    import ttnn

    from models.experimental.t0_forecast.tt.attention import sdpa_chunk
    from models.experimental.t0_forecast.tt.common import TtT0Precision, rms_norm, upload_norm

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(5)
    qkv = torch.randn(R * S, 3 * DM, generator=gen)
    ref = reference(qkv)

    device = ttnn.open_device(device_id=0)
    p = TtT0Precision.accuracy()
    kernel = p.compute_kernel_config(fp32_acc=True)
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gain = upload_norm(device, torch.ones(D))
    result = {"shape": {"B": B, "G": G, "S_pad": S, "H": H, "hd": D}}

    def heads(t):
        if len(t.shape) == 4:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                t, num_heads=H, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                t, num_heads=H, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        qn, kn = rms_norm(q, gain, 1e-8, p), rms_norm(k, gain, 1e-8, p)
        for x in (q, k):
            ttnn.deallocate(x)
        return qn, kn, v

    def sdpa(q, k, v, mask, chunk=None):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=chunk or sdpa_chunk(q.padded_shape[-2]),
            k_chunk_size=chunk or sdpa_chunk(k.padded_shape[-2]),
            exp_approx_mode=False,
        )
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=1 / math.sqrt(D),
            program_config=cfg,
            compute_kernel_config=kernel,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_concat_heads=True,
        )
        for x in (q, k, v):
            ttnn.deallocate(x)
        return out

    try:
        flat = up(qkv.reshape(1, 1, R * S, 3 * DM))

        def current():
            t = ttnn.reshape(ttnn.clone(flat), (R, S, 3 * DM))
            sm = ttnn.permute(t, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t)
            blocks = ttnn.reshape(sm, (S * B, G, 3 * DM))
            blocks = ttnn.fill_implicit_tile_padding(blocks, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v = heads(blocks)
            ttnn.deallocate(blocks)
            merged = ttnn.reshape(sdpa(q, k, v, None), (S, R, DM))
            rm = ttnn.permute(merged, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(merged)
            return ttnn.reshape(rm, (1, 1, R * S, DM))

        slot = torch.arange(G * S) % S

        def make_mask(fill, batch=1):
            m = torch.zeros(G * S, G * S).masked_fill(slot[:, None] != slot[None, :], fill)
            return up(m.reshape(1, 1, G * S, G * S).expand(batch, 1, -1, -1).contiguous())

        mask_dev = make_mask(float("-inf"))
        variant = {}

        def packed():
            t = ttnn.reshape(flat, (B, 1, G * S, 3 * DM))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                t, num_heads=H, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            qn, kn = rms_norm(q, gain, 1e-8, p), rms_norm(k, gain, 1e-8, p)
            for x in (q, k):
                ttnn.deallocate(x)
            return ttnn.reshape(sdpa(qn, kn, v, variant.get("mask", mask_dev), variant.get("chunk")), (1, 1, R * S, DM))

        def with_variant(**kw):
            def fn():
                variant.clear()
                variant.update(kw)
                return packed()

            return fn

        runs = [("current", current), ("packed", packed)]
        if args.diagnose:
            runs = [
                ("packed", packed),
                ("packed_neg1e4", with_variant(mask=make_mask(-1e4))),
                ("packed_batch_mask", with_variant(mask=make_mask(float("-inf"), B))),
                ("packed_chunk128", with_variant(chunk=128)),
                ("packed_chunk64", with_variant(chunk=64)),
            ]
        for name, fn in runs:
            try:
                out = ttnn.to_torch(fn()).float().reshape(R * S, DM)
                result[name] = {"pcc_vs_torch": pcc(ref, out), "max_abs": float((ref - out).abs().max())}
                per_b = [pcc(ref.reshape(B, -1)[b], out.reshape(B, -1)[b]) for b in range(B)]
                per_g = [pcc(ref.reshape(B, G, S, DM)[:, g], out.reshape(B, G, S, DM)[:, g]) for g in range(G)]
                result[name]["pcc_b_min_first_last"] = [min(per_b), per_b[0], per_b[-1]]
                result[name]["pcc_per_g"] = per_g
                result[name]["ms"] = time_op(device, fn) * 1e3
            except Exception as e:  # noqa: BLE001
                result[name] = {"error": str(e)[:400]}
            print(name, result[name], flush=True)
    finally:
        ttnn.close_device(device)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
