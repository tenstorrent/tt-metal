"""SDPA forward / backward wall time per call at the TP=4 shard shape (one chip), optionally the single-chip
model shape; `--ref` compares the outputs against a stored baseline (iter11_sdpa_check.py --save)."""
import argparse
import time

import numpy as np
import ttnn
import ttml

parser = argparse.ArgumentParser()
parser.add_argument("--ref")
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--cases", default="shard", help="shard | all")
parser.add_argument("--tag", default="")
args = parser.parse_args()

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
sdpa = ttml.ops.attention.scaled_dot_product_attention


def dev_t(a, requires_grad=False):
    t = ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(requires_grad)
    return t


def host(t):
    if hasattr(t, "get_value"):
        return t.to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def bench(fn, iters):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def report(name, got, ref):
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    diff = np.abs(got - ref)
    ndiff = int((diff > 0).sum())
    print(f"    {name}: pcc {pcc:.8f} max|diff| {diff.max():.3e} differing {ndiff}/{got.size}", flush=True)


CASES = [(8, 2, 1024, 128)]
if args.cases == "all":
    CASES = [
        (8, 2, 1024, 128),
        (32, 8, 1024, 128),
        (8, 8, 1024, 128),
        (4, 1, 1024, 128),
        (8, 2, 2048, 128),
        (6, 2, 512, 64),
    ]
ref = np.load(args.ref) if args.ref else None
for qh, kvh, S, D in CASES:
    key = f"{qh}_{kvh}_{S}_{D}"
    q_np = rng.normal(0, 1, (1, qh, S, D))
    k_np = rng.normal(0, 1, (1, kvh, S, D))
    v_np = rng.normal(0, 1, (1, kvh, S, D))
    dout_np = rng.normal(0, 1, (1, qh, S, D))
    q, k, v = dev_t(q_np, True), dev_t(k_np, True), dev_t(v_np, True)
    dout = dev_t(dout_np).get_value()

    def fw():
        return sdpa(q, k, v)

    def fw_bw():
        out = sdpa(q, k, v)
        out.set_grad(dout)
        out.backward(False)
        return out

    out = fw_bw()
    res = {"out": host(out), "dQ": host(q.get_grad()), "dK": host(k.get_grad()), "dV": host(v.get_grad())}
    t_fw = bench(fw, args.iters)
    t_fwbw = bench(fw_bw, args.iters)
    print(
        f"{args.tag} case q{qh} kv{kvh} S{S} D{D}: wall us fw {t_fw:.0f} fw+bw {t_fwbw:.0f} bw {t_fwbw - t_fw:.0f}",
        flush=True,
    )
    if ref is not None:
        for n in ("out", "dQ", "dK", "dV"):
            report(n, res[n], ref[f"{key}_{n}"])
ttml.autograd.AutoContext.get_instance().close_device()
