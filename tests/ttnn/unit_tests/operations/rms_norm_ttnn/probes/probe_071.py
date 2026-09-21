import ttnn
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=8192); print("mesh devices:", md.get_num_devices(), flush=True); ttnn.close_mesh_device(md); print("PROBE_OK")
