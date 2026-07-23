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


from ttnn.operations.rms_norm import rms_norm


def run(shape, ml, gamma=False, dt=ttnn.bfloat16):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    cfg = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device)
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    gt = None
    tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W).to(tdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    tag = str(ml).split(".")[-1]
    dtag = str(dt).split(".")[-1]
    print(
        f"{str(shape):20s} {tag:14s} g={int(gamma)} {dtag:9s}: PCC={p:.6f} maxdiff={md:.4f}  {'OK' if p>0.99 else '*** FAIL'}"
    )


W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
run((1, 1, 32, 64), W, gamma=True)
run((1, 1, 64, 128), W, gamma=True)  # HT_LOCAL=2
run((1, 1, 32, 50), W, gamma=True)  # W non-aligned, boundary core
run((1, 1, 32, 4096), W, gamma=True)  # per_w_t_padded=2, K=103, boundary
run((1, 1, 17, 50), W, gamma=True)  # both non-aligned
run((2, 4, 128, 512), W, gamma=True)  # HT_LOCAL=32, K=64
run((1, 32, 128), W, gamma=True)  # 3D
run((32, 64), W, gamma=True)  # 2D
run((1, 1, 32, 64), W, gamma=False, dt=ttnn.float32)
run((1, 1, 32, 4096), W, gamma=True, dt=ttnn.float32)
print("---- BLOCK ----")
run((1, 1, 256, 512), B, gamma=True)  # sh=26,sw=48 both sub-tile
run((2, 4, 128, 512), B, gamma=True)  # HT_LOCAL=4
run((1, 1, 64, 128), B, gamma=False)
ttnn.close_device(device)
