import ttnn

d = ttnn.open_device(device_id=0)
print("XY", [(x, str(d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)))) for x in range(12)])
print("XY", [(y, str(d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)))) for y in range(10)])
print(
    "DR",
    d.dram_grid_size(),
    [str(d.dram_core_from_logical_core(ttnn.CoreCoord(b, 0))) for b in range(d.dram_grid_size().x)],
)
ttnn.close_device(d)
