import torch, ttnn, traceback, sys
from ttnn.operations.rms_norm import rms_norm

sys.path.insert(0, "eval")
from eval.sharding import auto_shard_config, shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def ref(x, g, eps=1e-6):
    xf = x.float()
    r = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / r
    if g is not None:
        out = out * g.float().reshape(-1)
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


ML = ttnn.TensorMemoryLayout
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.float32}


def run(shape, ml, dt=ttnn.bfloat16, gdt=ttnn.bfloat16, glayout=ttnn.TILE_LAYOUT, mc=None, tag=""):
    try:
        if mc is None:
            mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
        t = torch.randn(*shape, dtype=TD[dt])
        ti = ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        gi = None
        g = None
        if gdt is not None:
            g = torch.randn(1, 1, 1, shape[-1], dtype=TD[gdt])
            gi = ttnn.from_torch(g, dtype=gdt, layout=glayout, device=device)
        o = ttnn.to_torch(rms_norm(ti, gamma=gi, memory_config=ti.memory_config())).float()
        e = ref(t, g)
        print(f"OK  {tag or shape} {str(ml).split('.')[-1]} shard={list(mc.shard_spec.shape)} pcc={pcc(o,e):.6f}")
    except Exception as ex:
        print(f"FAIL {tag or shape} {str(ml).split('.')[-1]}: {type(ex).__name__}: {str(ex)[:300]}")


# padding-tile case: W=8192 -> wg=256, per_w=3, ncores=86, 86*3=258 -> 2 padding tiles
run((1, 1, 32, 8192), ML.WIDTH_SHARDED, tag="pad-tiles-8192")
# w_non_aligned
run((1, 1, 32, 50), ML.WIDTH_SHARDED, tag="w50")
run((4, 8, 32, 47), ML.WIDTH_SHARDED, tag="w47")
run((1, 1, 32, 50), ML.BLOCK_SHARDED, tag="w50-blk")
# h non-aligned
run((1, 1, 17, 64), ML.WIDTH_SHARDED, tag="h17")
run((4, 8, 47, 256), ML.BLOCK_SHARDED, tag="h47-blk")
# no gamma
run((1, 1, 64, 128), ML.WIDTH_SHARDED, gdt=None, tag="nogamma")
run((1, 1, 64, 128), ML.BLOCK_SHARDED, gdt=None, tag="nogamma-blk")
# dtypes
run((1, 1, 64, 128), ML.WIDTH_SHARDED, dt=ttnn.float32, gdt=ttnn.float32, tag="fp32")
run((1, 1, 64, 128), ML.BLOCK_SHARDED, dt=ttnn.bfloat8_b, gdt=ttnn.bfloat8_b, tag="bfp8")
# RM gamma with tiled sharded activation
run((1, 1, 64, 128), ML.WIDTH_SHARDED, glayout=ttnn.ROW_MAJOR_LAYOUT, tag="rm-gamma")
# rank 2/3
run((1024, 1024), ML.WIDTH_SHARDED, tag="rank2")
run((2, 512, 1024), ML.BLOCK_SHARDED, tag="rank3")
# pinned perf geometries
for (h, w), ss, cg in [
    ((32, 1024), [32, 128], (8, 1)),
    ((32, 2304), [32, 256], (9, 1)),
    ((32, 5120), [32, 160], (8, 4)),
    ((32, 7168), [32, 256], (7, 4)),
]:
    mc = shard_config(ss, cg, ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    run((1, 1, h, w), ML.WIDTH_SHARDED, mc=mc, tag=f"perf-{h}x{w}")
mc = shard_config([1024, 128], (8, 8), ML.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
run((1, 1, 8192, 1024), ML.BLOCK_SHARDED, mc=mc, tag="perf-8192x1024")
ttnn.close_device(device)
