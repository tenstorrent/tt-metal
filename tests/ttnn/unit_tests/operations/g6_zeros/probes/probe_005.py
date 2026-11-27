import ttnn

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    print("type", type(dev), isinstance(dev, ttnn.MeshDevice))
    print("Device attr", hasattr(ttnn, "Device"), getattr(ttnn, "Device", None))
    g = dev.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    d = dev.dram_grid_size()
    print("dram grid", d.x, d.y)
    for name in ("num_dram_channels", "get_num_devices", "arch", "id", "get_device_ids", "is_initialized", "shape"):
        if hasattr(dev, name):
            try:
                print(name, getattr(dev, name)())
            except Exception as e:
                print(name, "err", e)
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([8, 12, 128, 128]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    print(
        "pages",
        t.buffer_num_pages(),
        "page",
        t.buffer_page_size(),
        "aligned",
        t.buffer_aligned_page_size(),
        "addr",
        t.buffer_address(),
    )
    ta = ttnn.TensorAccessorArgs(t)
    print("ct", list(ta.get_compile_time_args()))
    print("crt", list(ta.get_common_runtime_args()) if hasattr(ta, "get_common_runtime_args") else "n/a")
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        t2 = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 4, 256, 128]), dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        print(dt, t2.buffer_page_size(), t2.buffer_aligned_page_size())
    kd = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        compile_time_args=[1],
        runtime_args=[],
        common_runtime_args=[5, 6],
        config=ttnn.WriterConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
    k = pd.kernels[0]
    print("crt type", type(k.common_runtime_args))
    k.common_runtime_args[0] = 99
    print("after mutate via list elem", list(pd.kernels[0].common_runtime_args))
    pd.kernels[0].common_runtime_args[0] = 77
    print("after mutate chained", list(pd.kernels[0].common_runtime_args))
    ks = pd.kernels
    print("kernels type", type(ks))
    print("mem view", [m for m in dir(ttnn) if "memory_view" in m.lower() or "MemoryView" in m])
finally:
    ttnn.close_mesh_device(dev)
