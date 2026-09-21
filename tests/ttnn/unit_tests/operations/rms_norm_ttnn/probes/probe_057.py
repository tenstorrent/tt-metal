# Trivial: open and close a 1x4 mesh so the wrapper's post-run reset covers all four cards.
import ttnn
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=8192)
print("mesh devices:", md.get_num_devices(), flush=True)
ttnn.close_mesh_device(md)
print("PROBE_OK", flush=True)
