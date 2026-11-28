import ttnn

dev = ttnn.open_device(device_id=0)
try:
    print("PROBE arch", dev.arch())
    print("PROBE grid", dev.compute_with_storage_grid_size())
    print("PROBE dram_grid", dev.dram_grid_size())
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        try:
            print("PROBE optimal", noc, dev.get_optimal_dram_bank_to_logical_worker_assignment(noc))
        except Exception as e:
            print("PROBE optimal err", repr(e))
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([16, 12, 128, 128]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    print("PROBE f32", t.buffer_page_size(), t.buffer_aligned_page_size(), t.buffer_num_pages())
finally:
    ttnn.close_device(dev)
