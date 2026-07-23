import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
RM = ttnn.ROW_MAJOR_LAYOUT


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


def run(shape, gl, dt=ttnn.bfloat16, gdt=ttnn.bfloat16):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    cfg = auto_shard_config(list(shape), HEIGHT, layout=RM, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=RM, device=device, memory_config=cfg)
    tg = gt = None
    if gl is not None:
        W = shape[-1]
        gtdt = torch.float32 if gdt == ttnn.float32 else torch.bfloat16
        tg = torch.randn(W).to(gtdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=gdt, layout=gl, device=device)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    ok = p > 0.99 and out.memory_config().memory_layout == HEIGHT and out.layout == RM
    print(f"  {shape} gl={gl} {dt}/{gdt}: PCC={p:.6f} md={md:.5f} {'OK' if ok else 'FAIL'}")
    assert ok


# the 7 previously-OOM cases (fp32 W=8192)
run((1, 1, 32, 8192), RM, ttnn.float32, ttnn.float32)
run((1, 1, 32, 8192), RM, ttnn.float32, ttnn.bfloat16)
run((1, 32, 8192), RM, ttnn.float32, ttnn.float32)
run((128, 8192), RM, ttnn.float32, ttnn.float32)
run((128, 8192), None, ttnn.float32, None)
# regression spot checks
run((1, 1, 256, 512), RM)
run((1, 1, 32, 50), None)
run((1, 1, 17, 50), RM, ttnn.bfloat16, ttnn.float32)
run((4, 8, 32, 256), RM)
run((1, 1, 32, 8192), None)  # bf16 wide
ttnn.close_device(device)
print("ALL OK")
