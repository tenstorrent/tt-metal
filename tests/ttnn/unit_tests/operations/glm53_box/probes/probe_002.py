import torch, ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=24576)
print("RESULT mesh", mesh.shape, "devices", mesh.get_num_devices())
print("RESULT grid", mesh.compute_with_storage_grid_size())
for a in ("num_dram_channels", "dram_size_per_channel"):
    try:
        print("RESULT", a, getattr(mesh, a)())
    except Exception as e:
        print("RESULT", a, "n/a", type(e).__name__)
x = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 1, 32, 32)
t = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
)
for ax in (0, 1):
    g = ttnn.all_gather(t, dim=0, cluster_axis=ax, topology=ttnn.Topology.Linear)
    parts = ttnn.get_device_tensors(g)
    print("RESULT all_gather axis", ax, tuple(parts[0].shape), [ttnn.to_torch(p)[:, 0, 0, 0].tolist() for p in parts])
ttnn.close_mesh_device(mesh)
print("RESULT ok")
