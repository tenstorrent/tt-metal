import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.feature_spec import INPUTS
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import plan_summary

cache = _FabricMeshCache()
mesh = cache.get()
for (shape,) in INPUTS:
    for dtype in (ttnn.float32, ttnn.bfloat16):
        for ml in (
            ttnn.TensorMemoryLayout.INTERLEAVED,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ):
            try:
                mc = (
                    ttnn.DRAM_MEMORY_CONFIG
                    if ml == ttnn.TensorMemoryLayout.INTERLEAVED
                    else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=mesh)
                )
            except Exception as e:
                print("ERR shardcfg", shape, ml, type(e).__name__)
                continue
            if mc is None:
                print("ERR nocfg", shape, ml)
                continue
            rows, cols = tuple(mesh.shape)
            x = torch.zeros((rows, cols, *shape))
            glob = torch.cat([torch.cat(list(x[r]), dim=1) for r in range(rows)], dim=0)
            try:
                tx = ttnn.from_torch(
                    glob,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mc,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
                )
            except Exception as e:
                print("ERR alloc", shape, ml, str(e)[:80])
                continue
            g = (
                ttnn.from_torch(
                    torch.zeros(rows, cols * shape[-1])
                    .reshape(1, 1, rows, cols * shape[-1])[:, :, :1, :]
                    .repeat(1, 1, 1, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2, 3), mesh_shape=(1, cols)),
                )
                if False
                else None
            )
            try:
                p = plan_summary(tx, cluster_axis=1, gamma=None, num_links=1)
                print("PLAN", p["regime"])
            except NotImplementedError as e:
                print("REFUSE", shape, dtype, ml, str(e)[:120])
            except Exception as e:
                print("ERR plan", shape, ml, type(e).__name__, str(e)[:120])
            ttnn.deallocate(tx)
cache.close()
