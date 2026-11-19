import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.examples.fabric_all_gather import fabric_all_gather

c = _FabricMeshCache()
m = c.get((2, 2))
print("OPEN1 OK", m.shape)
x = ttnn.from_torch(
    torch.randn(64, 128),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=m,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(m, dims=(0, 1), mesh_shape=(2, 2)),
)
y = fabric_all_gather(x, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=1, dim=0)
ttnn.synchronize_device(m)
print("EXAMPLE OK", y.shape)
c.close()
print("CLOSED")
m = c.get((2, 2))
print("OPEN2 OK", m.shape)
c.close()
print("DONE")
