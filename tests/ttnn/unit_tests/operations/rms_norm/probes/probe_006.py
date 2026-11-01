import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
base = ttnn._ttnn.device.GetAllocatorBaseAddress(d, ttnn.BufferType.L1)
mv = ttnn.get_memory_view(d, ttnn.BufferType.L1) if hasattr(ttnn, "get_memory_view") else None
print("PROBE arch", d.arch(), "grid", g.x, g.y)
print("PROBE unreserved", ttnn.get_max_worker_l1_unreserved_size(), "alloc_base", base)
print("PROBE memview", mv)
try:
    print("PROBE l1_size", ttnn._ttnn.device.get_l1_size() if hasattr(ttnn._ttnn.device, "get_l1_size") else "n/a")
except Exception as e:
    print("PROBE l1_size err", e)
ttnn.close_device(d)
