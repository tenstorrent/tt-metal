import ttnn
d = ttnn.open_device(device_id=0)
print("arch", d.arch())
print("grid", d.compute_with_storage_grid_size())
for name in ("l1_size_per_core","l1_size","get_l1_size_per_core","l1_bank_size"):
    a = getattr(d, name, None)
    print(name, "->", a, "callable:", callable(a))
    if callable(a):
        try: print("   value", a())
        except Exception as e: print("   err", e)
print("clock", d.get_clock_rate_mhz() if hasattr(d,"get_clock_rate_mhz") else "?")
ttnn.close_device(d)
