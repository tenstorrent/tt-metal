# L1/DRAM bank counts behind the p1 bank-collision arithmetic and the BB-05 bound.
import ttnn

device = ttnn.open_device(device_id=0)
try:
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    print(f"PROBE l1_banks={l1.num_banks} dram_banks={dram.num_banks} grid={device.compute_with_storage_grid_size()}")
finally:
    ttnn.close_device(device)
