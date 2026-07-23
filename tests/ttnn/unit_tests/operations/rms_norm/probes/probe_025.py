import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


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


def run(shape, gamma, dt, gl):
    torch.manual_seed(0)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape).to(tdt)
    cfg = auto_shard_config(list(shape), HEIGHT, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
    print(f"  shard_spec: shape={cfg.shard_spec.shape} grid={cfg.shard_spec.grid}")
    xt = ttnn.from_torch(ti, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    gt = tg = None
    if gamma:
        W = shape[-1]
        tg = torch.randn(W).to(tdt)
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=dt, layout=gl, device=device)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = dt == ttnn.float32
    cc.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cc, memory_config=xt.memory_config())
    got = ttnn.to_torch(out)
    exp = ref(ti, tg)
    p = pcc(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    print(
        f"  {shape} gamma={gamma} {dt} gl={gl}: PCC={p:.6f} maxdiff={md:.5f} out_ml={out.memory_config().memory_layout}"
    )
    assert p > 0.99, "FAIL"
    print("  OK")


print("=== TILE input + TILE gamma ===")
run((1, 1, 256, 512), True, ttnn.bfloat16, ttnn.TILE_LAYOUT)
print("=== TILE input + no gamma ===")
run((1, 1, 256, 512), False, ttnn.bfloat16, ttnn.TILE_LAYOUT)
