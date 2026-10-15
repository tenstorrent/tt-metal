import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    di = ttnn._ttnn.reports.get_device_info(mesh)
    print(
        "info",
        di.l1_bank_size,
        di.address_at_first_l1_bank,
        di.worker_l1_size,
        ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1),
    )
finally:
    ttnn.close_mesh_device(mesh)
