import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if g is not None:
        o = o * g.float().reshape(-1)
    return o


def run(shape, ml, gamma=False, dt=ttnn.bfloat16):
    torch.manual_seed(0)
    ti = torch.randn(shape).to(torch.bfloat16)
    cfg = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    gt = None
    tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W).to(torch.bfloat16)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    from ttnn.operations.rms_norm import rms_norm

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    tag = str(ml).split(".")[-1]
    print(f"{shape} {tag} gamma={gamma}: PCC={p:.6f} maxdiff={md:.4f}  {'OK' if p>0.99 else 'FAIL'}")


run((1, 1, 32, 64), ttnn.TensorMemoryLayout.WIDTH_SHARDED, gamma=False)
ttnn.close_device(device)
