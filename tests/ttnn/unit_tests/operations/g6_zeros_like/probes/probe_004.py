import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        print(noc, [(c.x, c.y) for c in cores])
    print("dram grid", device.dram_grid_size())
    try:
        print("num dram channels", device.num_dram_channels())
    except Exception as e:
        print("no num_dram_channels", e)
    for y in range(g.y):
        print([str(device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))) for x in range(g.x)])
    x = ttnn.from_torch(
        torch.zeros((1, 12, 128, 128)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("pages", x.buffer_num_pages(), "aligned page", x.buffer_aligned_page_size(), "addr", hex(x.buffer_address()))
    print([m for m in dir(device) if "dram" in m.lower() or "bank" in m.lower()])
finally:
    ttnn.close_device(device)
