import torch, ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=24576)
d = mesh.get_devices()[0]
print("RESULT mesh", mesh.shape, "devices", mesh.get_num_devices())
print("RESULT grid", mesh.compute_with_storage_grid_size())
print("RESULT dram_banks", d.num_dram_channels(), "dram_bytes_per_bank", d.dram_size_per_channel())
x = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 1, 32, 32)
t = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
)
for dim_axis in (0, 1):
    g = ttnn.all_gather(t, dim=0, cluster_axis=dim_axis, topology=ttnn.Topology.Linear)
    print("RESULT all_gather axis", dim_axis, tuple(ttnn.get_device_tensors(g)[0].shape))
ttnn.close_mesh_device(mesh)
print("RESULT ok")
