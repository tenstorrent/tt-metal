import ttnn

device = ttnn.open_device(device_id=0)
print("GRID", device.dram_grid_size(), device.compute_with_storage_grid_size())
for ch in range(device.dram_grid_size().x):
    print("DRAM", ch, device.dram_core_from_logical_core(ttnn.CoreCoord(ch, 0)))
for y in range(10):
    print(
        "W ",
        y,
        device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)),
        device.worker_core_from_logical_core(ttnn.CoreCoord(10, y)),
    )
try:
    print("OPT", device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
except Exception as e:
    print("OPT err", e)
ttnn.close_device(device)
