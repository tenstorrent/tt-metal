import ttnn, torch

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    r = ttnn.split_work_to_cores(g, 1940, row_wise=True)
    print(r[0], r[4], r[5])
    print("g1", [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[2].ranges()], r[2].num_cores())
    print("g2", [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[3].ranges()], r[3].num_cores())
    r = ttnn.split_work_to_cores(g, 8, row_wise=True)
    print(
        r[0],
        r[4],
        r[5],
        [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[1].ranges()],
        [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[3].ranges()],
    )
    r = ttnn.split_work_to_cores(g, 130, row_wise=True)
    print(
        r[0],
        r[4],
        r[5],
        [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[1].ranges()],
        [(cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in r[2].ranges()],
    )
    cl = ttnn.grid_to_cores(13, g.x, g.y, True)
    print([(c.x, c.y) for c in cl])
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
    print("page", t.buffer_page_size(), t.buffer_num_pages(), t.buffer_aligned_page_size(), t.buffer_address())
    print(ttnn.TensorAccessorArgs(t).get_compile_time_args(), ttnn.TensorAccessorArgs().get_compile_time_args())
    tl = ttnn.from_torch(
        torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print(ttnn.TensorAccessorArgs(tl).get_compile_time_args())
    print(t.memory_config().memory_layout, t.memory_config().buffer_type, t.memory_config().is_sharded())
finally:
    ttnn.close_device(device)
