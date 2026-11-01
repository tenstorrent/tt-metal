import numpy as np, ttnn, ttml, sys

ttml.autograd.AutoContext.get_instance().open_device()
rng = np.random.default_rng(0)
sdpa = ttml.ops.attention.scaled_dot_product_attention
ref = np.load(sys.argv[1])


def dev_t(a, rg=False):
    t = ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(rg)
    return t


def host(t):
    if hasattr(t, "get_value"):
        return t.to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


CASES = [(8, 2, 1024, 128), (32, 8, 1024, 128), (8, 8, 1024, 128)]
for qh, kvh, S, D in CASES:
    key = f"{qh}_{kvh}_{S}_{D}"
    q_np = rng.normal(0, 1, (1, qh, S, D))
    k_np = rng.normal(0, 1, (1, kvh, S, D))
    v_np = rng.normal(0, 1, (1, kvh, S, D))
    dout_np = rng.normal(0, 1, (1, qh, S, D))
    dout = dev_t(dout_np).get_value()
    for rep in range(3):
        q, k, v = dev_t(q_np, True), dev_t(k_np, True), dev_t(v_np, True)
        out = sdpa(q, k, v)
        out.set_grad(dout)
        out.backward(False)
        for name, t in (("dV", v.get_grad()), ("dK", k.get_grad())):
            got = host(t)
            r = ref[f"{key}_{name}"]
            bad = np.argwhere(np.abs(got - r) > 0.5)
            print(f"{key} rep{rep} {name}: >0.5 diffs {len(bad)}", flush=True)
            if len(bad):
                heads = sorted(set(bad[:, 1]))
                rows = sorted(set(bad[:, 2] // 32))
                cols = sorted(set(bad[:, 3] // 32))
                print("   heads", heads, "row tiles", rows[:20], "col tiles", cols)
                b = bad[0]
                print("   example", b, "got", got[tuple(b)], "ref", r[tuple(b)])
ttml.autograd.AutoContext.get_instance().close_device()
