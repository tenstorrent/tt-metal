import ttnn, torch

device = ttnn.open_device(device_id=0)
try:
    print("dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y, "device id", device.id())
    n, all_c, g1, g2, p1, p2 = ttnn.split_work_to_cores(g, 2048, row_wise=True)
    print("split", n, p1, p2, "g1 cores", g1.num_cores(), "g2", g2.num_cores())
    print("all ranges", all_c, "g1", g1, "g2", g2)
    cores = ttnn.grid_to_cores(n, g.x, g.y, True)
    print("first cores", [(c.x, c.y) for c in cores[:14]])
    g1set = set((c.x, c.y) for c in ttnn.corerange_to_cores(g1)) if hasattr(ttnn, "corerange_to_cores") else None
    print("has corerange_to_cores", hasattr(ttnn, "corerange_to_cores"))
    if g1set is not None:
        print("g1 matches first", all(((c.x, c.y) in g1set) == (i < g1.num_cores()) for i, c in enumerate(cores)))
    t = ttnn.from_torch(
        torch.zeros(1, 8, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    print("idx page", t.buffer_page_size(), t.buffer_aligned_page_size(), t.storage_type())
    kd = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=all_c,
        compile_time_args=[1],
        runtime_args=[],
        common_runtime_args=[7, 8],
        config=ttnn.ReaderConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
    ks = pd.kernels
    print("type kernels", type(ks))
    ks[0].common_runtime_args[0] = 99
    print("after mutate via list", pd.kernels[0].common_runtime_args[0])
    pd.kernels[0].common_runtime_args[1] = 55
    print("after direct mutate", list(pd.kernels[0].common_runtime_args))
    kd.common_runtime_args[0] = 42
    print("kd mutated, pd sees", list(pd.kernels[0].common_runtime_args))
    print("uint16 dtype", ttnn.uint16, ttnn.uint32)
finally:
    ttnn.close_device(device)
