import ttnn, torch

dev = ttnn.open_device(device_id=0)
try:
    print("dram_grid_size", dev.dram_grid_size())
    try:
        a = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev)
        print("assignment len", len(a), a)
    except Exception as e:
        print("assign err", e)
    print("compute grid", dev.compute_with_storage_grid_size())
    print("dram align", ttnn.get_dram_alignment())
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(
        "page size",
        t.buffer_page_size(),
        "num pages",
        t.buffer().num_pages() if hasattr(t.buffer(), "num_pages") else "?",
    )
    b = t.buffer()
    print([n for n in dir(b) if "bank" in n.lower() or "page" in n.lower()])
    try:
        print("num_banks", b.num_banks())
    except Exception as e:
        print(e)
finally:
    ttnn.close_device(dev)
