import ttnn, math
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)


def show(shape, ml, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    cfg = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    ss = cfg.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    cores = ttnn.corerange_to_cores(ss.grid, None, True)
    K = len(cores)
    W = shape[-1]
    NC = 1
    for d in shape[:-2]:
        NC *= d
    H = shape[-2]
    print(
        f"shape={shape} {str(ml).split('.')[-1]} {str(dtype).split('.')[-1]}: sh={sh} sw={sw} K={K} bb={ss.grid.bounding_box()}"
    )
    print(
        f"   NC*H={NC*H} HT_LOCAL=ceil(sh/32)={math.ceil(sh/32)}  per_w_t_pad=ceil(sw/32)={math.ceil(sw/32)} sw%32={sw%32}"
    )
    vc = [min(sw, max(0, W - i * sw)) for i in range(K)]
    nb = sum(1 for v in vc if 0 < v < sw)
    print(f"   W={W} vc[:3]={vc[:3]} vc[-3:]={vc[-3:]} boundary(vc<sw)={nb}")


grid = device.compute_with_storage_grid_size()
print("compute grid:", grid, "=>", grid.x * grid.y, "cores")
for shape in [
    (1, 1, 32, 64),
    (1, 1, 64, 128),
    (1, 1, 32, 50),
    (1, 1, 32, 4096),
    (2, 4, 128, 512),
    (1, 1, 17, 50),
    (1, 32, 128),
    (32, 64),
]:
    show(shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
print("---- BLOCK ----")
for shape in [(1, 1, 256, 512), (2, 4, 128, 512), (1, 1, 64, 128)]:
    show(shape, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
print("---- fp32 WIDTH ----")
for shape in [(1, 1, 32, 64), (1, 1, 32, 4096)]:
    show(shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, dtype=ttnn.float32)

# --- shard-pad values probe: does from_torch zero the padded region of the last shard? ---
import torch

sh_shape = (1, 1, 32, 50)
t = torch.arange(1 * 1 * 32 * 50, dtype=torch.float32).reshape(sh_shape) + 1.0  # all nonzero
cfg = auto_shard_config(
    list(sh_shape),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    device=device,
)
xt = ttnn.from_torch(
    t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg
)
back = ttnn.to_torch(xt)
print("readback shape", back.shape, "matches?", torch.allclose(back.float(), t, atol=1))
ttnn.close_device(device)
