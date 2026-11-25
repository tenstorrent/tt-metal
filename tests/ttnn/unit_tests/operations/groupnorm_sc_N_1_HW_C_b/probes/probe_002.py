import ttnn

device = ttnn.open_device(device_id=0)
base = ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1)
print("alloc base", base, "unreserved", ttnn.get_max_worker_l1_unreserved_size())
print([a for a in dir(device) if "l1" in a.lower() or "size" in a.lower()])
try:
    print("l1_size_per_core", device.l1_size_per_core())
except Exception as e:
    print("err", e)
ttnn.close_device(device)
