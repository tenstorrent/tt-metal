import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config


def ref(x, g=None, eps=1e-6):
    xf = x.float()
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.float().reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).pow(2).mean().sqrt() / b.std()).item()


_ML = ttnn.TensorMemoryLayout
device = ttnn.open_device(device_id=0)
try:
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    CASES = [
        ((1, 1, 3232, 96), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 4064, 160), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((7136, 736), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 224, 3072), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 416, 1184), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 32, 72), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 224, 72), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 3232, 96), _ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 256, 512), _ML.HEIGHT_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 256, 512), _ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
        ((1, 1, 3232, 96), _ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
        ((7136, 736), _ML.BLOCK_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 3232, 96), None, ttnn.TILE_LAYOUT),
        ((1, 1, 224, 1000), None, ttnn.TILE_LAYOUT),
        ((1, 1, 333, 1000), None, ttnn.ROW_MAJOR_LAYOUT),
    ]
    for shape, ml, lay in CASES:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(shape[-1], dtype=torch.bfloat16)
        try:
            mc = (
                ttnn.DRAM_MEMORY_CONFIG
                if ml is None
                else auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=device)
            )
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
            tg = ttnn.from_torch(gm.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=lay, device=device)
            kw = {} if ml is None else {"memory_config": tx.memory_config()}
            got = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kw))
            e = ref(x, gm)
            print(
                f"{'INTERLEAVED' if ml is None else str(ml).split('.')[-1]:15s} {str(lay).split('.')[-1][:4]:5s} {str(shape):18s} PCC {pcc(got,e):.6f} relRMS {rms(got,e):.4f}",
                flush=True,
            )
        except Exception as ex:
            print(f"{ml} {lay} {shape} EXC {type(ex).__name__}: {str(ex)[:100]}", flush=True)
finally:
    ttnn.close_device(device)
