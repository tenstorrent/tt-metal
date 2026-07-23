import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


fails = []


def run(shape, dt, gdt):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    gtdt = torch.float32 if gdt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    cfg = auto_shard_config(list(shape), HEIGHT, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    W = shape[-1]
    tg = torch.randn(W).to(gtdt)
    gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=gdt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)  # RM gamma
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    ok = "OK" if p > 0.99 else "FAIL"
    if p <= 0.99:
        fails.append((shape, dt, gdt, p))
    print(f"  {shape} in={dt} RMgamma={gdt}: PCC={p:.6f} maxdiff={md:.4f} {ok}")


bf16 = ttnn.bfloat16
f32 = ttnn.float32
print("== TILE input + RM gamma ==")
run((1, 1, 256, 512), bf16, bf16)
run((1, 1, 256, 512), bf16, f32)  # mixed precision
run((1, 1, 32, 50), bf16, bf16)  # W non-aligned
run((1, 1, 50, 128), bf16, bf16)  # H non-aligned
run((1, 1, 17, 50), bf16, f32)  # both non-aligned, mixed
run((4, 8, 32, 256), bf16, bf16)  # multi-batch
run((1, 1, 32, 4096), bf16, bf16)  # wide, 1 core
run((128, 512), bf16, f32)  # 2D mixed
run((1, 1, 256, 512), f32, f32)  # fp32 input
print("FAILS:", fails)
ttnn.close_device(device)
