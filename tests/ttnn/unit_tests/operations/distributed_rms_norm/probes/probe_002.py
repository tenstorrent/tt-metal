import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm, plan_summary

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
for shape in ((1, 1, 640, 1792), (1, 1, 64, 256)):
    for ps in (False, True):
        x = torch.randn((4, *shape)).to(torch.bfloat16)
        g = torch.randn((4, 1, 1, 1, shape[-1])).to(torch.bfloat16)
        tx = ttnn.from_torch(
            torch.cat(list(x), dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        tg = ttnn.from_torch(
            torch.cat(list(g), dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        if ps:
            print("DBG plan", plan_summary(tx, cluster_axis=1, gamma=tg, topology=ttnn.Topology.Ring, num_links=2))
        out = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2)
        print(
            "DBG",
            shape,
            ps,
            [round(ttnn.to_torch(d).float().abs().max().item(), 3) for d in ttnn.get_device_tensors(out)],
        )
cache.close()
