import torch, ttnn

device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.zeros(1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.ones(1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print("EQ spec", a.spec == b.spec, a.spec == a.spec, hash(a.spec) == hash(b.spec), hash(a.spec) == hash(a.spec))
print("EQ shape", a.shape == b.shape, hash(a.shape) == hash(b.shape))
print("EQ memcfg", a.memory_config() == b.memory_config(), hash(a.memory_config()) == hash(b.memory_config()))
print("EQ dtype", hash(a.dtype) == hash(b.dtype), "layout", hash(a.layout) == hash(b.layout))
print("EQ dev id", a.device().id(), b.device().id())
ttnn.close_device(device)
