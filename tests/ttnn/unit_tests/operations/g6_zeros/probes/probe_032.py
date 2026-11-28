import ttnn

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
print("grid", dev.compute_with_storage_grid_size())
for noc in [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1]:
    cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev, noc)
    print(noc, [(c.x, c.y) for c in cores])
print("banks", ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).num_banks)
ttnn.close_mesh_device(dev)
