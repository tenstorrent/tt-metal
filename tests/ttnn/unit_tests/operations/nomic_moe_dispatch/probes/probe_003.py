import ttnn

d = ttnn.open_device(device_id=0)
print("ARCH", d.arch())
print("X", [d.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(11)])
print("Y", [d.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(10)])
ttnn.close_device(d)
