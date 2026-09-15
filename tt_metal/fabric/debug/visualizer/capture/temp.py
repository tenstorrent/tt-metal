import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    offset=ttnn.MeshCoordinate(0, 0),
)
print(f"opened OK: num_devices={mesh.get_num_devices()}")
ttnn.close_mesh_device(mesh)