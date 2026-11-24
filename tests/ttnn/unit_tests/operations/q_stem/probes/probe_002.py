import ttnn, types
from eval.golden_tests.q_stem.conftest import _FabricMeshCache, focus_mesh_or_skip
from eval.golden_tests.q_stem.feature_spec import INPUTS
from ttnn.operations.q_stem import q_stem_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(focus_mesh_or_skip())
grid = mesh.compute_with_storage_grid_size()
print("BUDGET", pd.cb_l1_budget(mesh))
T = lambda shape, dt=ttnn.bfloat16: types.SimpleNamespace(shape=shape, dtype=dt)
for x, qb, w, G in INPUTS:
    if (x[2] // G) % 32:
        continue
    geo = pd.Geometry(T(x), T(qb), T(w), G, grid, True)
    wkv_units_total_max = geo.hpc_max if geo.wkv_resident else geo.hpc_max * geo.rows_per_head * geo.n_nb
    wkv_depth = max(1, min(pd.WKV_DEPTH, wkv_units_total_max))
    t = pd.l1_terms(geo, wkv_depth)
    land = geo.landing_tiles * geo.tile_bytes
    print(
        "L1",
        x,
        qb[3],
        w[1:],
        "R1" if geo.wkv_resident else "R2",
        {k: sum(v.values()) // 1024 for k, v in t.items()},
        "landing KiB",
        land // 1024,
        "C1",
        geo.C1,
        "hpc",
        geo.hpc_max,
    )
cache.close()
