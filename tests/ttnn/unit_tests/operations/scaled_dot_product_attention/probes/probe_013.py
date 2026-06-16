import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def torch_ref(q, k, v, is_causal=True, scale=None):
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    s = (q @ k.transpose(-2, -1)) * scale
    S = q.shape[-2]
    Skv = k.shape[-2]
    if is_causal:
        m = torch.full((S, Skv), float("-inf")).triu(1)
        s = s + m
    return torch.softmax(s, -1) @ v


def run(device):
    torch.manual_seed(1234)
    b, h, s, d = 1, 1, 1024, 128
    Q = fa_rand(b, h, s, d)
    K = fa_rand(b, h, s, d)
    V = fa_rand(b, h, s, d)
    ref = torch_ref(Q.float(), K.float(), V.float(), is_causal=True)
    for label, fp32acc in [("fp32acc=True", True), ("fp32acc=False", False)]:
        if fp32acc:
            ckc = None
        else:
            ckc = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        tq = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tk = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=ckc)
        o = ttnn.to_torch(out).float()
        x = o.flatten()
        y = ref.flatten()
        pcc = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        rmse = (x - y).pow(2).mean().sqrt().item()
        print(f"bf8b {label}: PCC={pcc:.5f} rmse={rmse:.5f}")


device = ttnn.open_device(device_id=0)
try:
    run(device)
finally:
    ttnn.close_device(device)
