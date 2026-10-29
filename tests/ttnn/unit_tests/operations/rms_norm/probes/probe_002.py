import ttnn

dev = ttnn.open_device(device_id=0)
try:
    print("OUT", [n for n in dir(ttnn) if "Allocator" in n or "memory_view" in n.lower()])
    print("OUT base", ttnn.device.GetAllocatorBaseAddress(dev, ttnn.BufferType.L1))
    mv = ttnn.device.GetMemoryView(dev, ttnn.BufferType.L1)
    print("OUT mv", [a for a in dir(mv) if not a.startswith("_")])
    print("OUT", mv.total_bytes_per_bank if hasattr(mv, "total_bytes_per_bank") else None)
finally:
    ttnn.close_device(dev)
