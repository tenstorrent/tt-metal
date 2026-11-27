import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),
        ]
    )
    print("ranges order", [(r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges()])
    cores = ttnn.corerange_to_cores(grid, None, True)
    print("n cores", len(cores), cores[:3], cores[-2:])
    cores_c = ttnn.corerange_to_cores(grid, None, False)
    print("col-major", cores_c[:3], cores_c[-2:])
    print("l1 banks?", [a for a in dir(device) if "bank" in a.lower()])
    for bt in (ttnn.BufferType.L1, ttnn.BufferType.DRAM):
        try:
            print(bt, device.num_banks(bt))
        except Exception as e:
            print("num_banks err", e)
    spec = ttnn.ShardSpec(grid, (320, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    x = ttnn.from_torch(
        torch.randn(1, 128, 256, 32).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    print("padded", x.padded_shape, "page", x.buffer_page_size(), x.buffer_aligned_page_size(), x.buffer_num_pages())
    print(
        "shard spec",
        x.memory_config().shard_spec,
        x.memory_config().shard_spec.shape,
        x.memory_config().shard_spec.orientation,
    )
    print("is_sharded", x.is_sharded(), x.memory_config().is_sharded(), x.storage_type())
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, ttnn.L1_MEMORY_CONFIG)
    print("out page", out.buffer_page_size(), out.buffer_aligned_page_size(), out.buffer_num_pages())
    print("ta args", list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
    cb = ttnn.cb_descriptor_from_sharded_tensor(0, x)
    print(
        "cb",
        cb.total_size,
        cb.buffer_address(),
        x.buffer_address(),
        [(f.buffer_index, f.page_size) for f in cb.format_descriptors],
    )
    k = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=grid,
        compile_time_args=[1],
        runtime_args=[],
        common_runtime_args=[7],
        config=ttnn.WriterConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    pd.kernels[0].common_runtime_args = [9]
    print("mutated common rt", list(pd.kernels[0].common_runtime_args))
    kk = pd.kernels[0]
    kk.common_runtime_args = [11]
    print("mutated via ref", list(pd.kernels[0].common_runtime_args))
    c0 = pd.cbs[0]
    c0.set_buffer_from_cb(ttnn.cb_descriptor_from_sharded_tensor(0, out if False else x))
    print("cb has buffer", pd.cbs[0].has_buffer())
    print("dev id", device.id() if hasattr(device, "id") else None, type(device))
finally:
    ttnn.close_device(device)
