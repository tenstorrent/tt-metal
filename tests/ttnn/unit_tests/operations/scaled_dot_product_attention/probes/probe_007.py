import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[fin], b[fin]
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run(approx, dtype=ttnn.bfloat16, shape=(1, 2, 256, 64)):
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    s = 1.0 / math.sqrt(shape[-1])
    w = torch.softmax(torch.matmul(Qf, Kf.transpose(-2, -1)) * s, dim=-1)
    e = torch.matmul(w, Vf)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=approx
    )

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(dev(Q), dev(K), dev(V), compute_kernel_config=cfg)
    return pcc(e, ttnn.to_torch(out).float())


print(f"  math_approx_mode=False: PCC={run(False):.5f}")
print(f"  math_approx_mode=True : PCC={run(True):.5f}")
# also confirm a causal mask path (exp of -inf-derived large negatives) with approx
ttnn.close_device(device)
