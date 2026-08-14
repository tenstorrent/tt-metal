import ttnn
d = ttnn.open_device(device_id=0)
names = [n for n in dir(d) if "l1" in n.lower() or "bank" in n.lower() or "mem" in n.lower() or "alloc" in n.lower()]
print(names)
for n in names:
    a = getattr(d, n)
    try:
        print(n, "->", a() if callable(a) else a)
    except Exception as e:
        print(n, "ERR", type(e).__name__, str(e)[:80])
ttnn.close_device(d)
