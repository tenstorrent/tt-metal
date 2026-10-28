import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
shape = (1, 1, 64, 256)
x = torch.randn((4, *shape)).to(torch.bfloat16)
g = torch.randn((4, 1, 1, 1, 256)).to(torch.bfloat16)
for name, mapper in (
    ("sh1d", ttnn.ShardTensorToMesh(mesh, dim=-1)),
    ("sh2d", ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(1, 4))),
):
    if name == "sh1d":
        xin = torch.cat(list(x), dim=-1)
        gin = torch.cat(list(g), dim=-1)
    else:
        xin = (
            torch.cat(list(x), dim=1).reshape(1, 4, 64, 256) if False else torch.cat([torch.cat(list(x), dim=1)], dim=0)
        )
        gin = torch.cat(list(g), dim=1)
    tx = ttnn.from_torch(xin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
    tg = ttnn.from_torch(gin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
    print("DBG", name, list(tx.shape), list(tg.shape))
    for i, d in enumerate(ttnn.get_device_tensors(tx)):
        print("DBG x dev", i, ttnn.to_torch(d).float().abs().max().item())
    out = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2)
    for i, d in enumerate(ttnn.get_device_tensors(out)):
        print("DBG out dev", i, ttnn.to_torch(d).float().abs().max().item())
cache.close()
