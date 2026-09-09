import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for shape in ([], [64], [1, 1, 32, 50], [1, 1, 50, 50], [3, 50, 96], [1, 1, 1, 2048]):
        spec = ttnn.TensorSpec(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM, ttnn.Tile([32, 32])
        )
        t = ttnn.allocate_tensor_on_device(spec, device)
        print(
            "TILE out",
            shape,
            "logical",
            list(t.shape),
            "padded",
            list(t.padded_shape),
            "page",
            t.buffer_page_size(),
            "npages",
            t.buffer_num_pages(),
        )
        # RM input
        x = (
            torch.arange(max(1, int(torch.tensor(shape).prod()) if shape else 1)).reshape(shape).to(torch.bfloat16)
            if shape
            else torch.tensor(3.5, dtype=torch.bfloat16)
        )
        ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        print(
            "   RM in ",
            shape,
            "logical",
            list(ti.shape),
            "padded",
            list(ti.padded_shape),
            "page",
            ti.buffer_page_size(),
            "aligned",
            ti.buffer_aligned_page_size(),
            "npages",
            ti.buffer_num_pages(),
        )
finally:
    ttnn.close_device(device)
