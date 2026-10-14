import ttnn

m = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
for r in range(2):
    for c in range(2):
        print("COORD", r, c, m.get_device_id(ttnn.MeshCoordinate(r, c)))
ttnn.close_mesh_device(m)
