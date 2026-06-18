import ttnn

d = ttnn.open_device(device_id=0)
print("ARCH:", d.arch())
ttnn.close_device(d)
