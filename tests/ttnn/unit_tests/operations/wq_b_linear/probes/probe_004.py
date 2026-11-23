import ttnn

d = ttnn.open_device(device_id=0)
for n in dir(d):
    if "bank" in n.lower() or "grid" in n.lower():
        print(n)
try:
    print("L1 banks", d.num_banks(ttnn.BufferType.L1))
except Exception as e:
    print("err", e)
try:
    print("compute grid", d.compute_with_storage_grid_size())
except Exception as e:
    print("err", e)
ttnn.close_device(d)
