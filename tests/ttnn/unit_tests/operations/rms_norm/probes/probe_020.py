import torch, ttnn, traceback
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

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
dev = device
g = dev.compute_with_storage_grid_size()
print("GRID", g.x, g.y, ttnn.get_arch_name())
for shape in shapes:
    for layout, ln in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")):
        try:
            mc = auto_shard_config(list(shape), ML.HEIGHT_SHARDED, layout=layout, dtype=ttnn.bfloat16, device=dev)
            ss = mc.shard_spec
            ncores = ss.grid.num_cores()
            x = ttnn.from_torch(
                torch.randn(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=dev,
                memory_config=mc,
            )
            gm = ttnn.from_torch(
                torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=dev
            )
            out = rms_norm(x, gamma=gm, memory_config=mc)
            t = ttnn.to_torch(out)
            xr = torch.randn(1)  # noop
            print(f"OK   {shape} {ln} shard={list(ss.shape)} ncores={ncores}")
            del out
            del x, gm
        except Exception as e:
            msg = str(e).replace("\n", " ")[:220]
            print(f"FAIL {shape} {ln} shard={list(ss.shape)} ncores={ncores}: {type(e).__name__}: {msg}")
