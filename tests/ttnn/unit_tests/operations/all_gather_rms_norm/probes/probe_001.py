import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
x = ttnn.from_torch(
    torch.randn(1, 1, 64, 64),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
)
d1, d2 = x.device(), x.device()
print("PROBE same wrapper per call:", d1 is d2, id(d1) == id(d2), id(mesh) == id(d1))
for i in range(3):
    all_gather_rms_norm(x, cluster_axis=1, topology=ttnn.Topology.Ring)
    tabs = [k for k in pd._RES_CACHE if k[0] == "table"]
    print(
        "PROBE call",
        i,
        "cache entries",
        len(pd._RES_CACHE),
        "tables",
        len(tabs),
        "plans",
        len(pd._PLAN_CACHE),
        "table addr",
        [int(v[1].buffer_address()) for k, v in pd._RES_CACHE.items() if k[0] == "table"],
    )
y = ttnn.from_torch(
    torch.randn(1, 1, 640, 384),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
)
all_gather_rms_norm(y, cluster_axis=1, topology=ttnn.Topology.Ring)
print("PROBE after 2nd shape tables", [int(v[1].buffer_address()) for k, v in pd._RES_CACHE.items() if k[0] == "table"])
old_id = id(mesh)
cache.close()
mesh = cache.get((1, 4))
x = ttnn.from_torch(
    torch.randn(1, 1, 64, 64),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
)
print("PROBE reopened: same id as old mesh:", id(mesh) == old_id, id(x.device()) == old_id)
try:
    all_gather_rms_norm(x, cluster_axis=1, topology=ttnn.Topology.Ring)
    print("PROBE call after reopen OK")
except Exception as e:
    print("PROBE call after reopen FAILED:", str(e).splitlines()[0][:200])
cache.close()
