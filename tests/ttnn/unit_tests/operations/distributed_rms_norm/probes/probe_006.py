import ttnn, math
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.sharding import auto_shard_config
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get()
grid = mesh.compute_with_storage_grid_size()
print("PROBE grid", grid.x, grid.y, "budget", pd.l1_budget(mesh), "mesh", tuple(mesh.shape))


class Fake:
    def __init__(s, shape, mc):
        s.shape, s._mc = shape, mc

    def memory_config(s):
        return s._mc


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
]
ML = ttnn.TensorMemoryLayout
for shape in shapes:
    for ml in (ML.INTERLEAVED, ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED):
        for dt in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b):
            for acc in (True, False):
                if dt == ttnn.float32 and not acc:
                    continue
                for g in (None, ttnn.float32):
                    try:
                        mc = (
                            ttnn.DRAM_MEMORY_CONFIG
                            if ml == ML.INTERLEAVED
                            else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=mesh)
                        )
                    except Exception as e:
                        print("PROBE", shape, ml, dt, "shardfail", str(e)[:80])
                        continue
                    t = Fake(list(shape), mc)
                    G = 2
                    sz = pd.Sizes(
                        ttnn.tile_size(dt),
                        ttnn.tile_size(dt),
                        ttnn.tile_size(g) if g else 0,
                        ttnn.tile_size(pd.intermediate_format(acc)),
                        g is not None,
                        G,
                    )
                    for L in (1, 2):
                        taken = set()
                        if ml != ML.INTERLEAVED:
                            taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in pd._range_cores(r)}
                        try:
                            ports = pd.port_cores(grid, 2 * L, taken)
                        except Exception as e:
                            print("PROBE", shape, ml, dt, "ports", e)
                            continue
                        pset = {(c.x, c.y) for c in ports}
                        workers = [
                            ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset
                        ]
                        try:
                            p = pd.local_plan(t, workers, sz, pd.l1_budget(mesh), ml != ML.INTERLEAVED)
                        except Exception as e:
                            spec = mc.shard_spec
                            print(
                                "PROBE FAIL",
                                shape,
                                str(ml).split(".")[-1],
                                dt,
                                acc,
                                "g" if g else "-",
                                "L",
                                L,
                                None if spec is None else tuple(spec.shape),
                                str(e)[:110],
                            )
