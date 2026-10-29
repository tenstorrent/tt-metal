import ttnn

dev = ttnn.open_device(device_id=0)
try:
    print("OUT base", ttnn.device.get_allocator_base_address(dev, ttnn.BufferType.L1))
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    print("OUT mv", {a: getattr(mv, a) for a in dir(mv) if not a.startswith("_") and not callable(getattr(mv, a))})
finally:
    ttnn.close_device(dev)
