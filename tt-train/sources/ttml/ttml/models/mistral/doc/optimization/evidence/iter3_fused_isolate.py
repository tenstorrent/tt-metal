import numpy as np, ml_dtypes, torch, ttnn, ttml
from ttml.modules import LinearLayer, LoraConfig, LoraLinear, transpose_frozen_weights, fuse_lora_into_frozen_weights

ttml.autograd.AutoContext.get_instance().open_device()
rng = np.random.default_rng(0)
T, IN, OUT, R = 1024, 5120, 6144, 16


def bf(a):
    return a.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)


def dev(a, rg=False):
    t = ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(rg)
    return t


def host(t):
    return (
        t.to_numpy(ttnn.DataType.FLOAT32)
        if hasattr(t, "get_requires_grad")
        else ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)
    ).astype(np.float64)


def rel(a, b):
    return float(np.linalg.norm(a.ravel() - b.ravel()) / np.linalg.norm(b.ravel()))


x_np = bf(rng.normal(0, 1, (1, 1, T, IN)))
w_np = bf(rng.normal(0, 1 / np.sqrt(IN), (1, 1, OUT, IN)))
a_np = bf(rng.uniform(-1 / np.sqrt(IN), 1 / np.sqrt(IN), (1, 1, R, IN)))
b_np = bf(rng.normal(0, 0.02, (1, 1, OUT, R)))
g_np = bf(rng.normal(0, 0.01, (1, 1, T, OUT)))


def build():
    base = LinearLayer(IN, OUT, False)
    ttml.autograd.assign_numpy(base.weight.tensor, w_np.astype(np.float32))
    layer = LoraLinear(base, LoraConfig(rank=R, alpha=32.0))
    ttml.autograd.assign_numpy(layer.lora_A.tensor, a_np.astype(np.float32))
    ttml.autograd.assign_numpy(layer.lora_B.tensor, b_np.astype(np.float32))
    return layer


res = {}
for mode in ("ref", "fused"):
    layer = build()
    if mode == "fused":
        transpose_frozen_weights(layer)
        fuse_lora_into_frozen_weights(layer)
    x = dev(x_np, True)
    out = layer(x)
    o = host(out)
    out.set_grad(dev(g_np).get_value())
    out.backward(False)
    res[mode] = (o, host(x.get_grad()), host(layer.lora_A.tensor.get_grad()), host(layer.lora_B.tensor.get_grad()))
    ttml.autograd.AutoContext.get_instance().reset_graph()
    if mode == "fused":
        wc = host(layer.weight.tensor.get_value())
        print(
            "R: W_cat tail rows IN..IN+R vs 2*B^T rel",
            rel(wc[0, 0, IN : IN + R, :], 2 * b_np[0, 0].T),
            " rows beyond max",
            float(np.abs(wc[0, 0, IN + R :, :]).max()),
            " head vs W^T rel",
            rel(wc[0, 0, :IN, :], w_np[0, 0].T),
            flush=True,
        )
xr = torch.tensor(x_np, requires_grad=True)
ar = torch.tensor(a_np[0, 0], requires_grad=True)
br = torch.tensor(b_np[0, 0], requires_grad=True)
yr = xr @ torch.tensor(w_np[0, 0]).T + ((xr @ ar.T) @ br.T) * 2.0
yr.backward(torch.tensor(g_np))
for mode in ("ref", "fused"):
    o, dx, da, db = res[mode]
    print(
        f"R: {mode:5s} out {rel(o, yr.detach().numpy()):.2e}  dx {rel(dx, xr.grad.numpy()):.2e}  dA {rel(da[0,0], ar.grad.numpy()):.2e}  dB {rel(db[0,0], br.grad.numpy()):.2e}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
