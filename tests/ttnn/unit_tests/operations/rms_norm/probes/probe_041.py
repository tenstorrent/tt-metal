import ttnn, torch
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


device = ttnn.open_device(device_id=0)
try:
    WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    shape = (1, 1, 32, 64)
    torch.manual_seed(0)
    ti = torch.randn(shape).to(torch.bfloat16)
    cfg = auto_shard_config(list(shape), WIDTH, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    tg = torch.randn(shape[-1]).to(torch.bfloat16)
    gt = ttnn.from_torch(
        tg.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = False
    cc.math_approx_mode = False
    print(">>> running rms_norm with FORCED mcast across DRAM-col gap (grid 8x1) ...")
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    print(f">>> DONE. PCC={p:.6f} maxdiff={md:.6f}  {'PASS' if p>0.99 else 'FAIL'}")
finally:
    ttnn.close_device(device)
