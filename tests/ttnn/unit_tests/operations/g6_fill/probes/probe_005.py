import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(207),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("shape", x.shape, tuple(x.shape), len(x.shape), "padded", x.padded_shape)
    print("tile", x.tile.tile_shape, x.tile.face_shape)
    print("pages", x.buffer_num_pages(), "page", x.buffer_page_size(), "aligned", x.buffer_aligned_page_size())
    print("mc", x.memory_config(), x.memory_config().buffer_type, x.memory_config().memory_layout)
    print("on dev", ttnn.is_tensor_storage_on_device(x), x.storage_type())
    h = ttnn.from_torch(torch.randn(207), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    print("host on dev", ttnn.is_tensor_storage_on_device(h), h.storage_type())
    print("dev id", device.id(), x.device().id())
    grid = device.compute_with_storage_grid_size()
    print("grid", grid)
    crs = ttnn.num_cores_to_corerangeset(4, grid, True)
    print("crs4", crs)
    print("split crs", ttnn.split_work_to_cores(crs, 7, row_wise=True))
    print("split grid", ttnn.split_work_to_cores(grid, 16, row_wise=True))
    full = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, True)
    print("split full", ttnn.split_work_to_cores(full, 16, row_wise=True))
    r = ttnn.split_work_to_cores(full, 16, row_wise=True)[1]
    print([(c.x, c.y) for c in ttnn.corerange_to_cores(r, None, True)][:20])
    print(
        "rm",
        ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        ).buffer_page_size(),
    )
    print("es", x.element_size())
finally:
    ttnn.close_device(device)
