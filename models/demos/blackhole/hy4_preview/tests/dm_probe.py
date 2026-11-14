"""probe: open 1x8 row, all_gather_async dim2, all_to_all_async_generic semantics"""
import os
import sys

import torch

import ttnn

sys.path.insert(0, "/home/container_app_user/tt-metal")
from models.demos.gpt_oss.tt.ccl import CCLManager

topo = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}[os.environ.get("TOPO", "ring")]
ttnn.set_fabric_config(
    {"ring": ttnn.FabricConfig.FABRIC_1D_RING, "linear": ttnn.FabricConfig.FABRIC_1D}[os.environ.get("TOPO", "ring")]
)
if os.environ.get("OPEN", "sub") == "sub":
    full = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=100_000_000)
    md = full.create_submesh(ttnn.MeshShape(1, 8), ttnn.MeshCoordinate(0, 0))
else:
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
print("MESH", md.shape, flush=True)
ccl = CCLManager(md, num_links=2, topology=topo)
mm = ttnn.ShardTensorToMesh(md, dim=0)
S = 2
# per-device tensor [1,8,8S,16] value = 1000*dev + 100*h + row
W = 512
x = torch.zeros(8, 8, 8 * S, W)
for d in range(8):
    for h in range(8):
        for r in range(8 * S):
            x[d, h, r, :] = 1000 * d + 100 * h + r
LAY = {"rm": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}[os.environ.get("LAY", "rm")]
xd = ttnn.from_torch(
    x, device=md, dtype=ttnn.float32, layout=LAY, mesh_mapper=mm, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
cat = ttnn.ConcatMeshToTensor(md, dim=0)
for in_dim, out_dim in [(3, 2), (0, 2), (1, 2)]:
    try:
        y = ttnn.experimental.all_to_all_async_generic(
            xd, in_dim=in_dim, out_dim=out_dim, num_links=2, topology=topo, cluster_axis=1
        )
        yt = ttnn.to_torch(y, mesh_composer=cat)
        print("A2A", in_dim, out_dim, tuple(y.shape), tuple(yt.shape), flush=True)
        # device 3's view
        v = yt.reshape(8, *y.shape)[3]
        print(
            " dev3 first-col values (flattened over leading dims, then col blocks of W):",
            v.reshape(-1, v.shape[-1])[:, ::W][:, :].flatten().tolist()[:48],
            flush=True,
        )
    except Exception as e:
        print("A2A FAIL", in_dim, out_dim, str(e).strip().splitlines()[0][:200], flush=True)
if os.environ.get("OPEN", "sub") == "sub":
    ttnn.close_mesh_device(md)
    ttnn.close_mesh_device(full)
else:
    ttnn.close_mesh_device(md)
