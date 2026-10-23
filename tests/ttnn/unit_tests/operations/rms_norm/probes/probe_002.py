import ttnn

d = ttnn.open_device(device_id=0)
try:
    print("base", ttnn._ttnn.device.GetAllocatorBaseAddress(d, ttnn.BufferType.L1))
except Exception as e:
    print("err", e)
try:
    mv = ttnn._ttnn.device.GetMemoryView(d, ttnn.BufferType.L1)
    print("attrs", [a for a in dir(mv) if not a.startswith("_")])
    print("mv", {a: getattr(mv, a) for a in dir(mv) if not a.startswith("_") and not callable(getattr(mv, a))})
except Exception as e:
    print("err2", e)
print("l1", [a for a in dir(ttnn._ttnn.device) if "emory" in a or "Alloc" in a])
ttnn.close_device(d)
