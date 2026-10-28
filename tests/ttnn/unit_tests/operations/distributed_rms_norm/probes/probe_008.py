import ttnn, math
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.sharding import auto_shard_config
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get()
grid = mesh.compute_with_storage_grid_size()
shapes = [
    (1, 1, 32, 4096),
    (1, 32, 4096),
    (1, 1, 128, 4096),
    (1, 1, 32, 8192),
    (128, 8192),
    (32, 2048),
    (1, 32, 1792),
    (640, 1792),
    (1, 1, 4096, 1024),
    (1, 2048, 1280),
    (1, 1, 2048, 1280),
    (4, 1, 512, 512),
    (1, 1, 2048, 2048),
    (1, 1, 640, 1792),
]
ML = ttnn.TensorMemoryLayout
for shape in shapes:
    for ml in (ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED):
        for dt in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b):
            for acc in (True, False):
                if dt == ttnn.float32 and not acc:
                    continue
                for g in (None, ttnn.float32):
                    try:
                        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=mesh)
                    except Exception as e:
                        continue
                    try:
                        tin = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, mesh, mc)
                    except Exception as e:
                        print("PROBE", shape, str(ml).split(".")[-1], dt, "INPUT OOM")
                        continue
                    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, mesh, mc)
                    budget = pd.l1_budget(mesh)
                    G = 2
                    sz = pd.Sizes(
                        ttnn.tile_size(dt),
                        ttnn.tile_size(dt),
                        ttnn.tile_size(g) if g else 0,
                        ttnn.tile_size(pd.intermediate_format(acc)),
                        g is not None,
                        G,
                    )
                    L = 2
                    taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in pd._range_cores(r)}
                    try:
                        ports = pd.port_cores(grid, 2 * L, taken)
                        pset = {(c.x, c.y) for c in ports}
                        workers = [
                            ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset
                        ]
                        p = pd.local_plan(tin, workers, sz, budget, True)
                        ok = "ok"
                    except Exception as e:
                        ok = str(e)[:90]
                    spec = mc.shard_spec
                    if ok != "ok":
                        print(
                            "PROBE",
                            shape,
                            str(ml).split(".")[-1],
                            dt,
                            acc,
                            "g" if g else "-",
                            "shard",
                            tuple(spec.shape),
                            "ncores",
                            spec.grid.num_cores(),
                            "budget",
                            budget,
                            ok,
                        )
                    tin.deallocate()
                    tout.deallocate()
