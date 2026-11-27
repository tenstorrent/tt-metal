import torch, ttnn

device = ttnn.open_device(device_id=0)
x = ttnn.from_torch(
    torch.randn(207), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
print("type dev", type(device), hasattr(device, "id"), device.id() if hasattr(device, "id") else None)
d = x.device()
print("x.device()", type(d), d is device, d == device, d.id())
print(
    "shape",
    tuple(x.shape),
    "padded",
    tuple(x.padded_shape),
    "pages",
    x.buffer_num_pages(),
    "page size",
    x.buffer_page_size(),
)
print("tile", x.tile.tile_shape, x.tile.face_shape, x.element_size())
mc = x.memory_config()
print(mc, mc.buffer_type, mc.memory_layout, hash(mc) if hasattr(mc, "__hash__") else None)
print("grid", device.compute_with_storage_grid_size())
print(ttnn.is_tensor_storage_on_device(x))
h = ttnn.from_torch(torch.randn(207), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
print("host on dev", ttnn.is_tensor_storage_on_device(h))
try:
    print("host device()", h.device())
except Exception as e:
    print("host device() raised", type(e), e)
print(ttnn.TensorAccessorArgs(x).get_compile_time_args())
cores = ttnn.num_cores_to_corerangeset(16, device.compute_with_storage_grid_size(), True)
print(cores)
print(ttnn.split_work_to_cores(cores, 7, True))
print(ttnn.corerange_to_cores(cores, None, True)[:13])
ttnn.close_device(device)
