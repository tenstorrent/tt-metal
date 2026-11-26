import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y, type(device))
    print("arch", device.arch())
    for name in ["GetMemoryView", "get_memory_view"]:
        print(name, hasattr(ttnn, name), hasattr(ttnn.device, name) if hasattr(ttnn, "device") else None)
    try:
        mv = ttnn.device.GetMemoryView(device, ttnn.BufferType.L1)
        print("L1 banks", mv.num_banks, mv.total_bytes_per_bank)
    except Exception as e:
        print("mv err", e)
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
    print("spec eq", h.tensor_spec() == h.tensor_spec(), h.tensor_spec() == h2.tensor_spec())
    print(
        "page size",
        h.buffer_page_size(),
        h.buffer_aligned_page_size(),
        h.memory_config().buffer_type,
        h.storage_type(),
        h.is_allocated(),
    )
    print("mesh shape", getattr(device, "shape", None))
    try:
        print(device.shape[0], device.shape.dims() if hasattr(device.shape, "dims") else None)
    except Exception as e:
        print("shape err", e)
    # descriptor mutation probe
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
    print("rt before", pd.kernels[0].runtime_args[0][0])
    pd.kernels[0].runtime_args[0][0][1] = 99
    print("rt after elem set", pd.kernels[0].runtime_args[0][0])
    pd.kernels[0].runtime_args[0][0] = [7, 8, 9]
    print("rt after row set", pd.kernels[0].runtime_args[0][0])
    k0 = pd.kernels[0]
    k0.runtime_args[0][0] = [4, 5, 6]
    print("rt via k0", pd.kernels[0].runtime_args[0][0], k0.runtime_args[0][0])
    kl = pd.kernels
    print(
        type(kl),
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
    print(
        "tap",
        t.shape,
        t.padded_shape,
        t.logical_volume() if hasattr(t, "logical_volume") else t.volume(),
        t.buffer_num_pages(),
    )
finally:
    ttnn.close_device(device)
