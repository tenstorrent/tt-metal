import ttnn, torch

device = ttnn.open_device(device_id=0)
try:
    print("dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y, device.arch())
    r = ttnn.split_work_to_cores(g, 1940, row_wise=True)
    print(r[0], r[4], r[5], r[1])
    cores = [c for c in r[2]]
    print(len(cores), cores[:3], cores[-2:])
    print(type(r[1]), [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[1].ranges()])
    kd = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=r[1],
        compile_time_args=[1],
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[1, 2, 3],
        config=ttnn.ReaderConfigDescriptor(),
    )
    kd.common_runtime_args = [4, 5, 6]
    print("crta", list(kd.common_runtime_args))
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
    pd.custom_program_hash = 12345
    print("hash", pd.custom_program_hash)
    k0 = pd.kernels[0]
    k0.common_runtime_args = [7]
    print("after copy mutate", list(pd.kernels[0].common_runtime_args))
    t = ttnn.from_torch(
        torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    print("page", t.buffer_page_size(), t.buffer_num_pages(), t.buffer_aligned_page_size())
    print(ttnn.TensorAccessorArgs(t).get_compile_time_args(), ttnn.TensorAccessorArgs().get_compile_time_args())
    print(ttnn.grid_to_cores(3, g.x, g.y, True))
finally:
    ttnn.close_device(device)
