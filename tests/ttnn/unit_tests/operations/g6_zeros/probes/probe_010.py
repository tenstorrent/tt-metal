import ttnn

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    print("type", type(device), isinstance(device, ttnn.MeshDevice))
    print("grid", device.compute_with_storage_grid_size())
    mv = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    print("mv", mv, "num_banks", mv.num_banks)
    print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
    g = device.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(104, g, True)
    print("crs", crs, crs.num_cores())
    print("cores", [(c.x, c.y) for c in ttnn.corerange_to_cores(crs, None, True)][:20])
    print("id", device.id() if hasattr(device, "id") else None)
    print("dir", [a for a in dir(device) if "open" in a.lower() or "close" in a.lower() or "id" == a or "shape" in a])
    print(
        "opt",
        device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)
        if hasattr(device, "get_optimal_dram_bank_to_logical_worker_assignment")
        else "n/a",
    )
    print("ttnn.Device?", hasattr(ttnn, "Device"), hasattr(ttnn, "IDevice"))
finally:
    ttnn.close_mesh_device(device)
