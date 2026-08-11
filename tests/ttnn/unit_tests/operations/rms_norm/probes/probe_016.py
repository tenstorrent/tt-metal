import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout


def ref(x, g=None, eps=1e-6):
    x = x.float()
    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False

cases = []
# pad_poison shapes (TILE, poisoned implicit tile padding) x 3 sharded schemes
for shape in [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 224, 72), (1, 1, 40, 40)]:
    for ml in (ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        cases.append((shape, ml, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, None))
# bfloat8_b + fp32 sharded
for dt in (ttnn.bfloat8_b, ttnn.float32):
    for ml in (ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        cases.append(((1, 1, 256, 512), ml, ttnn.TILE_LAYOUT, dt, False, None))
# the pinned perf geometries
for rows, W, sh, grid, ml in [
    (32, 1024, [32, 128], (8, 1), ML.WIDTH_SHARDED),
    (32, 2304, [32, 256], (9, 1), ML.WIDTH_SHARDED),
    (32, 5120, [32, 160], (8, 4), ML.WIDTH_SHARDED),
    (32, 7168, [32, 256], (7, 4), ML.WIDTH_SHARDED),
    (8192, 1024, [1024, 128], (8, 8), ML.BLOCK_SHARDED),
]:
    cases.append(((1, 1, rows, W), ml, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, (sh, grid)))
try:
    for shape, ml, layout, dt, poison, pin in cases:
        tag = f"RESULT {str(shape):18s} {str(ml).split('.')[-1]:16s} {str(dt).split('.')[-1]:10s} {'poison' if poison else ('pinned' if pin else '      ')}"
        try:
            tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
            x = torch.randn(shape, dtype=tdt)
            g = torch.randn(shape[-1], dtype=tdt)
            mc = (
                shard_config(pin[0], pin[1], ml, layout=layout, dtype=dt, device=device)
                if pin
                else auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
            )
            tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=mc)
            tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dt, layout=layout, device=device)
            if poison:
                tx = ttnn.fill_implicit_tile_padding(tx, 1000.0)
                tg = ttnn.fill_implicit_tile_padding(tg, 1000.0)
            out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=tx.memory_config())
            got = ttnn.to_torch(out)
            exp = ref(x, g)
            print(
                f"{tag} PCC={pcc(got, exp):.6f} ratio={float((got.float().abs().sum()+1e-9)/(exp.abs().sum()+1e-9)):.4f} shard={list(mc.shard_spec.shape)}"
            )
        except Exception as e:
            print(f"{tag} FAILED {type(e).__name__}: {str(e)[:150]}")
finally:
    ttnn.close_device(device)
