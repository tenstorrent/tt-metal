import ttnn

# Test with all parameters
mesh_shape = ttnn.MeshShape(2, 2)
physical_device_ids = [0, 1, 2, 3]
offset = ttnn.MeshCoordinate(0, 0)  # Optional offset
dispatch_core_config = ttnn.DispatchCoreConfig()

device = ttnn.open_mesh_device(
    mesh_shape=mesh_shape,
    l1_small_size=ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size=ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    num_command_queues=1,
    dispatch_core_config=dispatch_core_config,
    offset=offset,
    physical_device_ids=physical_device_ids,
    worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
)

print(f"Successfully opened mesh device with shape: {mesh_shape}")
print(f"Physical device IDs: {physical_device_ids}")
