import ttnn

device = ttnn.open_device(device_id=0)
mv = ttnn.get_memory_view(device, ttnn.BufferType.L1)
print([a for a in dir(mv) if not a.startswith("_")])
for a in dir(mv):
    if not a.startswith("_"):
        try:
            print(a, getattr(mv, a))
        except Exception as e:
            print(a, e)
ttnn.close_device(device)
