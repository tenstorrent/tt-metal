import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
g = m.compute_with_storage_grid_size()
print("GRID", g.x, g.y)
d = m.get_devices()[0]
for lc in [(0, 0), (5, 0), (0, 3), (g.x - 1, 0)]:
    print("GRID logical", lc, "virtual", d.worker_core_from_logical_core(ttnn.CoreCoord(*lc)))
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
