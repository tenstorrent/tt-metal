import torch, ttnn

device = ttnn.open_device(device_id=0)
for x in [torch.tensor(1.5), torch.randn(64), torch.randn(50)]:
    try:
        t = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        print(
            "R0",
            list(t.shape),
            list(t.padded_shape),
            t.buffer_page_size(),
            t.buffer_aligned_page_size(),
            t.element_size(),
        )
    except Exception as e:
        print("ERR", str(e)[:300])
ttnn.close_device(device)
