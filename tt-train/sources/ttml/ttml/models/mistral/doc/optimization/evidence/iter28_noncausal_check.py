"""Review round 10 hard-check gap: the SDPA kernels' non-chain path (an explicit mask: USE_ATTN_MASK, work-list /
standard modes) at 8-row forward chunks. The explicit mask is the causal one, so the causal chain path of the same
inputs is the reference (same math, a different kernel path; one bf16 ulp expected from the chunking)."""
import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
rng = np.random.default_rng(0)
sdpa = ttml.ops.attention.scaled_dot_product_attention


def dev_t(a, rg=False):
    t = ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(rg)
    return t


def host(t):
    if hasattr(t, "get_value"):
        return t.to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def report(name, got, ref):
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    diff = np.abs(got - ref)
    print(
        f"    {name}: pcc {pcc:.8f} max|diff| {diff.max():.3e} differing {int((diff > 0).sum())}/{got.size}", flush=True
    )


ok = True
for qh, kvh, S, D in [(8, 2, 1024, 128), (6, 2, 512, 64)]:
    q_np = rng.normal(0, 1, (1, qh, S, D))
    k_np = rng.normal(0, 1, (1, kvh, S, D))
    v_np = rng.normal(0, 1, (1, kvh, S, D))
    dout = dev_t(rng.normal(0, 1, (1, qh, S, D))).get_value()
    mask_np = np.tril(np.ones((S, S), np.float32))[None, None]  # 1 = keep
    outs = {}
    for label, use_mask in (("causal kernel (chain)", False), ("explicit causal mask (non-chain)", True)):
        q, k, v = dev_t(q_np, True), dev_t(k_np, True), dev_t(v_np, True)
        out = sdpa(q, k, v, dev_t(mask_np).get_value()) if use_mask else sdpa(q, k, v)
        out.set_grad(dout)
        out.backward(False)
        outs[label] = {"out": host(out), "dQ": host(q.get_grad()), "dK": host(k.get_grad()), "dV": host(v.get_grad())}
    print(f"case q{qh} kv{kvh} S{S} D{D}: explicit mask path vs the causal chain path", flush=True)
    for n in ("out", "dQ", "dK", "dV"):
        report(n, outs["explicit causal mask (non-chain)"][n], outs["causal kernel (chain)"][n])
        pcc = np.corrcoef(
            outs["explicit causal mask (non-chain)"][n].ravel(), outs["causal kernel (chain)"][n].ravel()
        )[0, 1]
        ok &= pcc > 0.9999
print("NONCAUSAL PATH " + ("PASS" if ok else "FAIL"), flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
