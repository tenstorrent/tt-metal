import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(None)
TSL = ttnn.TensorMemoryLayout
cases = [
    ((1, 1, 32, 4096), TSL.HEIGHT_SHARDED, ttnn.bfloat16, True, True),
    ((1, 1, 32, 4096), TSL.HEIGHT_SHARDED, ttnn.bfloat16, True, False),
    ((1, 1, 640, 1792), TSL.HEIGHT_SHARDED, ttnn.float32, True, True),
    ((1, 1, 2048, 2048), TSL.WIDTH_SHARDED, ttnn.float32, True, False),
    ((1, 1, 4096, 1024), TSL.WIDTH_SHARDED, ttnn.bfloat16, True, False),
    ((1, 2048, 1280), TSL.WIDTH_SHARDED, ttnn.float32, True, True),
]
for shape, ml, dt, fp32, has_g in cases:
    mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=mesh)
    x = ttnn.from_torch(
        torch.randn(2 * shape[0] if len(shape) > 0 else 1, *shape[1:])
        if False
        else torch.randn((2 * shape[0],) + tuple(shape[1:-1]) + (2 * shape[-1],)),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, len(shape) - 1), mesh_shape=(2, 2)),
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), dt, ttnn.TILE_LAYOUT, mesh, mc)
    g = (
        ttnn.from_torch(
            torch.randn(1, 1, 1, 2 * shape[-1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=(2, 2)),
        )
        if has_g
        else None
    )
    budget = pd.l1_budget(mesh) - pd.LANDING_L1_RESERVE
    T = pd.tensor_tile_rows_of(x)
    Wt = -(-shape[-1] // 32)
    work, S, rows_max = pd._sharded_work(x, T, Wt)
    G_w = max(w.index for w in work) + 1
    geo = pd.compute_geometry(x, g, pd.W_CHUNK_TILES, fp32, S, True, G_w > 1, G_w, (shape[-1] % 32) != 0, 2)
    inv = pd.cb_inventory(geo, 1, 1, dt, dt, ttnn.bfloat16)
    big = sorted(((p * n, idx) for idx, _, p, n in inv), reverse=True)[:4]
    print(
        "PROBE",
        shape,
        str(ml).split(".")[-1],
        dt,
        "gamma" if has_g else "nog",
        "shard",
        list(mc.shard_spec.shape),
        "cores",
        len(work),
        "rows_max",
        rows_max,
        "S",
        S,
        "G_w",
        G_w,
        "budget",
        budget,
        "fp(R=1)",
        pd.footprint(geo, 1, 1),
        "fp(R=rows)",
        pd.footprint(geo, rows_max, 1),
        "top",
        big,
    )
    ttnn.deallocate(out)
    ttnn.deallocate(x)
cache.close()
