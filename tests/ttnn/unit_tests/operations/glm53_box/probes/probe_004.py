import torch, ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
print("RESULT grid", mesh.compute_with_storage_grid_size())
x = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 1, 32, 32)
t = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
)
g = ttnn.all_gather(t, dim=0, cluster_axis=1, topology=ttnn.Topology.Linear)
print("RESULT 1x4 all_gather", [ttnn.to_torch(p)[:, 0, 0, 0].tolist() for p in ttnn.get_device_tensors(g)])
ttnn.close_mesh_device(mesh)
