import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for name in ["get_dram_alignment", "get_l1_alignment"]:
        for mod in [ttnn, getattr(ttnn, "device", None), ttnn._ttnn.device]:
            f = getattr(mod, name, None) if mod is not None else None
            if f is not None:
                print("ALIGN", mod.__name__, name, f())
                break
    for shape, dt in [
        ((5,), ttnn.bfloat16),
        ((1, 1), ttnn.float32),
        ((1, 5120), ttnn.bfloat16),
        ((1, 1, 1, 62080), ttnn.bfloat16),
        ((3, 40, 50), ttnn.int32),
    ]:
        t = ttnn.from_torch(
            torch.zeros(shape).to(torch.int32 if dt == ttnn.int32 else torch.float32),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(
            "SHAPE",
            shape,
            dt,
            "logical",
            list(t.shape),
            "padded",
            list(t.padded_shape),
            "pages",
            t.buffer_num_pages(),
            "aps",
            t.buffer_aligned_page_size(),
        )
finally:
    ttnn.close_device(device)
