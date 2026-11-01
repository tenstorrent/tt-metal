import ttnn

d = ttnn.open_device(device_id=0)
mv = ttnn.get_memory_view(d, ttnn.BufferType.L1)
print("PROBE fields", [a for a in dir(mv) if not a.startswith("_")])
for a in dir(mv):
    if not a.startswith("_"):
        print("PROBE", a, getattr(mv, a))
x = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 32 * 130, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, d, ttnn.L1_MEMORY_CONFIG
)
mv = ttnn.get_memory_view(d, ttnn.BufferType.L1)
print(
    "PROBE after L1 tensor",
    mv.largest_contiguous_bytes_free_per_bank if hasattr(mv, "largest_contiguous_bytes_free_per_bank") else mv,
)
ttnn.close_device(d)
