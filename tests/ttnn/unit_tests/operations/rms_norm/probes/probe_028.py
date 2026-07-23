import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
grid = device.compute_with_storage_grid_size()
print("grid:", grid.x, grid.y, "=", grid.x * grid.y)


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


fails = []


def run(shape, gamma, dt):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    try:
        cfg = auto_shard_config(list(shape), HEIGHT, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
    except Exception as e:
        print(f"  {shape} {dt} g={gamma}: shardcfg SKIP {e}")
        return
    ss = cfg.shard_spec
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    gt = tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W).to(tdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
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
        fails.append((shape, dt, gamma, p))
    print(
        f"  {shape} {dt} g={gamma}: shard={list(ss.shape)} ncores={len(ttnn.corerange_to_cores(ss.grid,None,True))} PCC={p:.6f} maxdiff={md:.4f} {ok}"
    )


bf16 = ttnn.bfloat16
f32 = ttnn.float32
bf8 = ttnn.bfloat8_b
print("== aligned, per_h>1 (R>grid) ==")
run((1, 1, 8192, 256), True, bf16)  # R=256 -> per_h up to 3
run((1, 1, 8192, 512), False, bf16)  # R=256, wider W
print("== aligned small ==")
run((4, 8, 32, 256), True, bf16)  # NC=32, R=32
run((2, 4, 128, 512), True, bf16)  # R=32
print("== W non-aligned ==")
run((1, 1, 32, 50), True, bf16)
run((1, 1, 64, 17), True, bf16)
run((2, 1, 128, 100), False, bf16)
print("== H non-aligned ==")
run((1, 1, 50, 128), True, bf16)
run((4, 8, 47, 256), True, bf16)
print("== both non-aligned ==")
run((1, 1, 17, 50), True, bf16)
print("== fp32 ==")
run((1, 1, 256, 512), True, f32)
run((1, 1, 50, 128), True, f32)
print("== bf8b tile-aligned ==")
run((1, 1, 256, 512), True, bf8)
run((2, 4, 128, 512), False, bf8)
print("== 3D / 2D ==")
run((4, 128, 512), True, bf16)
run((128, 512), True, bf16)
run((1024, 1024), False, bf16)
print("== wide W ==")
run((1, 1, 32, 4096), True, bf16)
run((1, 1, 64, 8192), True, bf16)
print("FAILS:", fails)
ttnn.close_device(device)
