# Hypothesis probe: monkeypatch the program descriptor's acc_dtype decision by
# testing fp32_acc=False with fp32/bf8b inputs under the CURRENT code (acc=bf16
# intermediates) vs a patched version (fp32 intermediates). We test current first.
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


def run(dtype, acc):
    shape = (1, 2, 128, 64)
    td = _TD[dtype]
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=td)
    K = torch.randn(shape, dtype=td)
    V = torch.randn(shape, dtype=td)
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    s = 1.0 / math.sqrt(shape[-1])
    w = torch.softmax(torch.matmul(Qf, Kf.transpose(-2, -1)) * s, dim=-1)
    e = torch.matmul(w, Vf)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
    )

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(dev(Q), dev(K), dev(V), compute_kernel_config=cfg)
    return pcc(e, ttnn.to_torch(out).float())


print("=== CURRENT code (acc_dtype follows fp32_dest_acc_en) ===")
for d in [ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat16]:
    print(f"  {str(d).split('.')[-1]:9s} fp32_acc=False: PCC={run(d,False):.5f}   fp32_acc=True: PCC={run(d,True):.5f}")
ttnn.close_device(device)
