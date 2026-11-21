import torch, ttnn
device = ttnn.open_device(device_id=0)
for mc in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
    for W in (50, 64, 197, 4096):
        t = ttnn.from_torch(torch.zeros(1,1,32,W), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=mc)
        print(mc.buffer_type, W, t.buffer_page_size(), t.buffer_aligned_page_size(), t.buffer_num_pages())
print("l1 align", ttnn.get_l1_alignment(), "dram align", ttnn.get_dram_alignment())
ttnn.close_device(device)
