import ttnn
import time
import os

os.environ["TT_METAL_LOGGER_LEVEL"] = "DEBUG"  # Enable debug logging

print("Creating mesh device...")
mesh_device = ttnn.create_mesh_device(ttnn.MeshShape(2, 4), mesh_type=ttnn.MeshType.Ring)

print("Waiting 2 seconds...")
time.sleep(2)

print("Closing mesh device...")
ttnn.close_mesh_device(mesh_device)

print("Waiting 5 seconds for cleanup...")
time.sleep(5)

print("Done!")
