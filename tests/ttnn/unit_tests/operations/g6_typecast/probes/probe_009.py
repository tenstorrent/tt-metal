import ttnn

d = ttnn.open_device(device_id=0)
for y in range(10):
    c = d.worker_core_from_logical_core(ttnn.CoreCoord(0, y))
    c2 = d.worker_core_from_logical_core(ttnn.CoreCoord(10, y))
    print("L", y, c, c2)
print("N", ttnn.get_memory_view(d, ttnn.BufferType.DRAM).num_banks)
for n in dir(d):
    if "dram" in n.lower():
        print("D", n)
ttnn.close_device(d)
