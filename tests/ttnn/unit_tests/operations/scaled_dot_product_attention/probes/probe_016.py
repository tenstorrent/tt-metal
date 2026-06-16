import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def pcc(a, b):
    x = a.flatten().float()
    y = b.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-12)


torch.manual_seed(1234)
b, h, s, d = 1, 1, 1024, 128
Q = fa_rand(b, h, s, d)
K = fa_rand(b, h, s, d)
V = fa_rand(b, h, s, d)
ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), is_causal=True)
dev = ttnn.open_device(device_id=0)
try:
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=ckc)
    o = ttnn.to_torch(out).float()
    print(
        f"bf8b fp16-DEST: PCC={float(pcc(o,ref)):.5f} rmse={(o.flatten()-ref.flatten()).pow(2).mean().sqrt().item():.5f}"
    )
finally:
    ttnn.close_device(dev)
