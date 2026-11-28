# Bank counts behind the L1-budget bound and the p1 bank-collision arithmetic.
import ttnn

device = ttnn.open_device(device_id=0)
try:
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    print(f"BANKS l1={l1.num_banks} dram={dram.num_banks} grid={device.compute_with_storage_grid_size()}")
finally:
    ttnn.close_device(device)
