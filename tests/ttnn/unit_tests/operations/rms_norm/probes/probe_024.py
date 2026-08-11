import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
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
shapes = [(1, 224, 11008), (1, 1, 160, 11008)]
try:
    for shape in shapes:
        for layout, ln in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")):
            x = gm = out = None
            try:
                mc = auto_shard_config(
                    list(shape), ML.HEIGHT_SHARDED, layout=layout, dtype=ttnn.bfloat16, device=device
                )
                t = torch.randn(shape, dtype=torch.bfloat16)
                gt = torch.randn(shape[-1], dtype=torch.bfloat16)
                x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
                gm = ttnn.from_torch(gt.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
                out = rms_norm(x, gamma=gm, memory_config=mc)
                got = ttnn.to_torch(out)
                print(f"MSG OK   {shape} {ln} shard={list(mc.shard_spec.shape)} PCC={pcc(got, ref(t, gt)):.6f}")
            except Exception as e:
                print(f"MSG FAIL {shape} {ln}: {type(e).__name__}: {str(e)[:200]}")
            finally:
                for tt_ in (out, x, gm):
                    if tt_ is not None:
                        ttnn.deallocate(tt_)
finally:
    ttnn.close_device(device)
