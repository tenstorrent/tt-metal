import ttnn

device = ttnn.open_device(device_id=0)
info = ttnn._ttnn.reports.get_device_info(device)
print(
    "PROBE l1_num_banks",
    info.l1_num_banks,
    "l1_bank_size",
    info.l1_bank_size,
    "per storage",
    info.num_banks_per_storage_core,
    "grid",
    device.compute_with_storage_grid_size(),
)
print("PROBE arch", device.arch())
ttnn.close_device(device)
