import ttnn

d = ttnn.open_device(device_id=0)
print("arch", d.arch(), "grid", d.compute_with_storage_grid_size())
for n in dir(d):
    if "l1" in n.lower() or "alloc" in n.lower() or "size" in n.lower():
        print(n)
try:
    print("l1_size_per_core", d.l1_size_per_core())
except Exception as e:
    print(e)
try:
    print("allocator base", d.allocator().get_base_allocator_addr(ttnn.HalMemType.L1))
except Exception as e:
    print("e", e)
print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
ttnn.close_device(d)
