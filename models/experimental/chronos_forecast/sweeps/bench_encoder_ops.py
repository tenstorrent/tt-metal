# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Trace-timed microbenchmarks of the encoder's non-matmul ops at one L1 chunk (64 series x 160 tokens),
with the dtypes and memory configs of TtChronosPrecision.performance().

    python models/experimental/chronos_forecast/sweeps/bench_encoder_ops.py [rope add rms_norm heads qkv_rope concat]
"""

import sys

import torch
import ttnn

from models.experimental.chronos_forecast import ops
from models.experimental.chronos_forecast.sweeps.sweep_matmul_l1 import pcc, trace_us

SERIES, H, T, DH = 64, 12, 133, 64
D = H * DH
L1 = ttnn.L1_MEMORY_CONFIG


def to_dev(t, dtype, dev, mem=L1):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def bench_rope(dev):
    x = to_dev(torch.randn(SERIES, H, T, DH), ttnn.bfloat8_b, dev)
    cos = to_dev(torch.randn(1, 1, T, DH), ttnn.bfloat16, dev)
    sin = to_dev(torch.randn(1, 1, T, DH), ttnn.bfloat16, dev)
    x_h, cos_h, sin_h = (ttnn.to_torch(t).float() for t in (x, cos, sin))
    ref = x_h * cos_h + rotate_half(x_h) * sin_h
    for name, fn in (
        ("rope ttnn", lambda: ttnn.experimental.rotary_embedding(x, cos, sin, memory_config=L1)),
        ("rope chronos op", lambda: ops.rotary_embedding(x, cos, sin, memory_config=L1)),
    ):
        out = fn()
        got = ttnn.to_torch(out).float()[..., :T, :]
        ttnn.deallocate(out)
        yield name, trace_us(dev, fn), pcc(ref, got)


def bench_add(dev):
    a = to_dev(torch.randn(SERIES, T, D), ttnn.bfloat16, dev)
    b = to_dev(torch.randn(SERIES, T, D), ttnn.bfloat8_b, dev)
    ref = ttnn.to_torch(a).float() + ttnn.to_torch(b).float()
    cases = [("add bf16+bf8 ttnn", lambda: ttnn.add(a, b, memory_config=L1))]
    cases += [
        (
            f"add chronos {'bank' if local else 'rows'} b={n}",
            lambda n=n, local=local: ops.add(a, b, memory_config=L1, batch=n, bank_local=local),
        )
        for local in (False, True)
        for n in (2, 4, 8)
    ]
    for name, fn in cases:
        out = fn()
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        yield name, trace_us(dev, fn), pcc(ref, got)


def bench_rms_norm(dev):
    x = to_dev(torch.randn(SERIES, T, D), ttnn.bfloat16, dev)
    g_h = torch.rand(D) + 0.5
    g = ttnn.from_torch(g_h.reshape(1, 1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x_h = ttnn.to_torch(x).float()
    normed = x_h * torch.rsqrt(x_h.pow(2).mean(-1, keepdim=True) + 1e-6)
    for name, weight, ref in (("rms_norm gamma(DRAM)", g, normed * g_h), ("rms_norm no gamma", None, normed)):
        fn = lambda weight=weight: ttnn.rms_norm(x, epsilon=1e-6, weight=weight, memory_config=L1)
        out = fn()
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        yield name, trace_us(dev, fn), pcc(ref, got)


def bench_add_rms_norm(dev):
    """Residual add + RMSNorm: stock norm, model-local norm (bf16 / bf8 out), and the fused dual-output op."""
    x = to_dev(torch.randn(SERIES, T, D), ttnn.bfloat16, dev)
    out = to_dev(torch.randn(SERIES, T, D), ttnn.bfloat8_b, dev)
    y_h = ttnn.to_torch(x).float() + ttnn.to_torch(out).float()
    ref = y_h * torch.rsqrt(y_h.pow(2).mean(-1, keepdim=True) + 1e-6)

    def split(norm):
        def fn():
            y = ops.add(x, out, memory_config=L1)
            return y, norm(y)

        return fn

    def stock(y):
        return ttnn.rms_norm(y, epsilon=1e-6, memory_config=L1)

    def local(dtype):
        return lambda y: ops.rms_norm(y, epsilon=1e-6, memory_config=L1, norm_dtype=dtype)

    def fused(dtype):
        return lambda: ops.add_rms_norm(x, out, epsilon=1e-6, memory_config=L1, norm_dtype=dtype)

    cases = [
        ("add + ttnn.rms_norm", split(stock)),
        ("add + ops.rms_norm bf16", split(local(ttnn.bfloat16))),
        ("add + ops.rms_norm bf8", split(local(ttnn.bfloat8_b))),
        ("add_rms_norm bf16", fused(ttnn.bfloat16)),
        ("add_rms_norm bf8", fused(ttnn.bfloat8_b)),
    ]
    for name, fn in cases:
        y, n = fn()
        p = min(pcc(y_h, ttnn.to_torch(y).float()), pcc(ref, ttnn.to_torch(n).float()))
        ttnn.deallocate(y)
        ttnn.deallocate(n)

        def keep_n(fn=fn):
            y, n = fn()
            ttnn.deallocate(y)
            return n

        yield name, trace_us(dev, keep_n), p


def bench_heads(dev):
    xqkv = to_dev(torch.randn(SERIES, T, 3 * D), ttnn.bfloat8_b, dev)

    def fn():
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            xqkv, num_heads=H, transpose_key=False, memory_config=L1
        )
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return q

    ref = ttnn.to_torch(xqkv).float()[..., :D].reshape(SERIES, T, H, DH).permute(0, 2, 1, 3)
    out = fn()
    got = ttnn.to_torch(out).float()[..., :T, :]
    ttnn.deallocate(out)
    yield "create_qkv_heads", trace_us(dev, fn), pcc(ref, got)


def bench_qkv_rope(dev):
    xqkv = to_dev(torch.randn(SERIES, T, 3 * D), ttnn.bfloat8_b, dev)
    cos = to_dev(torch.randn(1, 1, T, DH), ttnn.bfloat16, dev)
    sin = to_dev(torch.randn(1, 1, T, DH), ttnn.bfloat16, dev)
    x_h, cos_h, sin_h = (ttnn.to_torch(t).float() for t in (xqkv, cos, sin))
    q_h, k_h, v_h = (p.reshape(SERIES, T, H, DH).permute(0, 2, 1, 3) for p in x_h.split(D, dim=-1))
    refs = [q_h * cos_h + rotate_half(q_h) * sin_h, k_h * cos_h + rotate_half(k_h) * sin_h, v_h]

    def split_then_rope():
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            xqkv, num_heads=H, transpose_key=False, memory_config=L1
        )
        out = [ops.rotary_embedding(t, cos, sin, memory_config=L1) for t in (q, k)] + [v]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return out

    def fused():
        return list(ops.qkv_heads_rope(xqkv, cos, sin, num_heads=H, memory_config=L1))

    for name, fn in (("split + 2x rope", split_then_rope), ("qkv_heads_rope", fused)):
        outs = fn()
        p = min(pcc(ref, ttnn.to_torch(t).float()[..., :T, :]) for ref, t in zip(refs, outs))
        for t in outs:
            ttnn.deallocate(t)

        def keep_q(fn=fn):
            q, *rest = fn()
            for t in rest:
                ttnn.deallocate(t)
            return q

        yield name, trace_us(dev, keep_q), p


def bench_concat(dev):
    ctx = to_dev(torch.randn(SERIES, H, T, DH), ttnn.bfloat8_b, dev)
    ref = ttnn.to_torch(ctx).float().permute(0, 2, 1, 3).reshape(SERIES, T, D)
    fn = lambda: ttnn.transformer.concatenate_heads(ctx, memory_config=L1)
    out = fn()
    got = ttnn.to_torch(out).float()[:, :T]
    ttnn.deallocate(out)
    yield "concat_heads", trace_us(dev, fn), pcc(ref, got)


BENCHES = {
    "rope": bench_rope,
    "add": bench_add,
    "rms_norm": bench_rms_norm,
    "add_rms_norm": bench_add_rms_norm,
    "heads": bench_heads,
    "qkv_rope": bench_qkv_rope,
    "concat": bench_concat,
}


def main():
    names = sys.argv[1:] or list(BENCHES)
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
    try:
        for name in names:
            for label, us, p in BENCHES[name](dev):
                print(f"{label:24s} {us:8.1f} us  pcc={p:.5f}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
