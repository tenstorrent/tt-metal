import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
_TD = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    fin = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[fin], b[fin]
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run(dtype, fid, acc, dist):
    shape = (1, 2, 128, 64)
    td = _TD[dtype]
    torch.manual_seed(0)
    g = torch.randn if dist == "randn" else torch.rand
    Q = g(shape, dtype=td)
    K = g(shape, dtype=td)
    V = g(shape, dtype=td)
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    s = 1.0 / math.sqrt(shape[-1])
    w = torch.softmax(torch.matmul(Qf, Kf.transpose(-2, -1)) * s, dim=-1)
    e = torch.matmul(w, Vf)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=acc, math_approx_mode=False)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(dev(Q), dev(K), dev(V), compute_kernel_config=cfg)
    got = ttnn.to_torch(out).float()
    return pcc(e, got)


worst = {}
for dtype in [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]:
    for fid in [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi]:
        for acc in [True, False]:
            for dist in ["randn", "rand"]:
                p = run(dtype, fid, acc, dist)
                key = (str(dtype).split(".")[-1], str(fid).split(".")[-1])
                worst[key] = min(worst.get(key, 1.0), p)
ttnn.close_device(device)
print("WORST PCC per (dtype, fidelity) over acc x dist:")
for k in sorted(worst):
    print(f"  {k[0]:9s} {k[1]:6s}: {worst[k]:.5f}")
