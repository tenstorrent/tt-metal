import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm


def pcc(a, b):
    x = a.float().flatten()
    y = b.float().flatten()
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def ref(x, g):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * g.float().reshape(-1)


device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
# WIDTH shard with a per-core hidden slice too wide to hold => chunking WITH a
# real cross-core combine (G = 2), and BLOCK likewise (G = 2 per grid row).
cases = [
    ((1, 1, 32, 16384), [32, 8192], (2, 1), ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT),
    ((1, 1, 32, 16384), [32, 8192], (2, 1), ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 1, 64, 16384), [32, 8192], (2, 2), ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT),
]
try:
    for shape, ss, grid, ml, layout in cases:
        x = gm = out = None
        try:
            mc = shard_config(ss, grid, ml, layout=layout, dtype=ttnn.bfloat16, device=device)
            t = torch.randn(shape, dtype=torch.bfloat16)
            gt = torch.randn(shape[-1], dtype=torch.bfloat16)
            x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
            gm = ttnn.from_torch(gt.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
            out = rms_norm(x, gamma=gm, memory_config=mc)
            got = ttnn.to_torch(out)
            print(f"MSG OK   {shape} {ml} {layout} PCC={pcc(got, ref(t, gt)):.6f}")
        except Exception as e:
            print(f"MSG FAIL {shape} {ml} {layout}: {type(e).__name__}: {str(e)[:200]}")
        finally:
            for tt_ in (out, x, gm):
                if tt_ is not None:
                    ttnn.deallocate(tt_)
finally:
    ttnn.close_device(device)
