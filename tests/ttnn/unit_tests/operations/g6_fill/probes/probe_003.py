import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    print("arch", device.arch())
    for n in [1, 3, 4, 5, 512, 513, 1000]:
        t = ttnn.from_torch(
            torch.zeros(n, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(
            n,
            "page",
            t.buffer_page_size(),
            "aligned",
            t.buffer_aligned_page_size(),
            "pages",
            t.buffer_num_pages(),
            "elsz",
            t.element_size(),
            "addr",
            hex(t.buffer_address()),
        )
finally:
    ttnn.close_device(device)
