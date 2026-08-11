import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
shapes = [
    (1, 1, 32, 4064),
    (1, 1, 96, 6144),
    (1, 1, 160, 11008),
    (1, 1, 992, 3000),
    (3, 1, 736, 5119),
    (1, 1, 32, 4095),
    (100, 5120),
    (1, 224, 11008),
    (3104, 4064),
]
g = device.compute_with_storage_grid_size()
print("MSG GRID", g.x, g.y, ttnn.get_arch_name())
try:
    for shape in shapes:
        for layout, ln in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")):
            ss = None
            try:
                mc = auto_shard_config(
                    list(shape), ML.HEIGHT_SHARDED, layout=layout, dtype=ttnn.bfloat16, device=device
                )
                ss = list(mc.shard_spec.shape)
                nc = mc.shard_spec.grid.num_cores()
                x = ttnn.from_torch(
                    torch.randn(shape, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=layout,
                    device=device,
                    memory_config=mc,
                )
                gm = ttnn.from_torch(
                    torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=layout,
                    device=device,
                )
                out = rms_norm(x, gamma=gm, memory_config=mc)
                t = ttnn.to_torch(out)
                print(f"MSG OK   {shape} {ln} shard={ss} ncores={nc}")
                del out, x, gm
            except Exception as e:
                print(f"MSG FAIL {shape} {ln} shard={ss}: {type(e).__name__}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)
