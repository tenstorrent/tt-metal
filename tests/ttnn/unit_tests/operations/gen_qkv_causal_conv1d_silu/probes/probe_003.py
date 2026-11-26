import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    mv = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print("L1 banks", mv.num_banks, mv.total_bytes_per_bank)
    h = ttnn.from_torch(
        torch.randn(1, 3, 96).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    h2 = ttnn.from_torch(
        torch.randn(1, 3, 96).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print("spec eq", h.spec == h.spec, h.spec == h2.spec)
    print(
        "page size",
        h.buffer_page_size(),
        h.buffer_aligned_page_size(),
        h.memory_config().buffer_type,
        h.storage_type(),
        h.is_allocated(),
    )
    print("mesh shape", device.shape, device.shape[0])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [1, 2, 3]
    kd = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
    print("rt before", list(pd.kernels[0].runtime_args[0][0]))
    try:
        pd.kernels[0].runtime_args[0][0][1] = 99
    except Exception as e:
        print("elem set err", e)
    print("rt after elem set", list(pd.kernels[0].runtime_args[0][0]))
    pd.kernels[0].runtime_args[0][0] = [7, 8, 9]
    print("rt after row set", list(pd.kernels[0].runtime_args[0][0]))
    k0 = pd.kernels[0]
    k0.runtime_args[0][0] = [4, 5, 6]
    print("rt via k0", list(pd.kernels[0].runtime_args[0][0]), list(k0.runtime_args[0][0]))
    print(
        type(pd.kernels),
        type(pd.kernels[0].runtime_args),
        type(pd.kernels[0].runtime_args[0]),
        type(pd.kernels[0].runtime_args[0][0]),
    )
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    print(cfg.math_fidelity, cfg.dst_full_sync_en)
    t = ttnn.from_torch(torch.randn(1, 1, 96).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("tap", t.shape, t.padded_shape, t.logical_volume(), t.buffer_num_pages())
    print(ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG.is_sharded(), ttnn.DRAM_MEMORY_CONFIG.memory_layout)
finally:
    ttnn.close_device(device)
