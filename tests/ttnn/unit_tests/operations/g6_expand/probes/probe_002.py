import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, 512).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    d = x.device()
    print("dev type", type(d))
    print("has id", hasattr(d, "id"), d.id() if hasattr(d, "id") else None)
    g = d.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    print("on dev", ttnn.is_tensor_storage_on_device(x))
    h = ttnn.from_torch(torch.randn(1, 1, 1, 512).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("host on dev", ttnn.is_tensor_storage_on_device(h), h.storage_type())
    mc = x.memory_config()
    print(mc.buffer_type, mc.memory_layout, hash(mc) if hasattr(mc, "__hash__") else None)
    print("page", x.buffer_page_size(), x.buffer_aligned_page_size(), x.padded_shape, x.shape)
    k = ttnn.KernelDescriptor(
        kernel_source="void kernel_main(){}",
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        compile_time_args=[1],
        common_runtime_args=[5],
        config=ttnn.ReaderConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[k])
    ks = pd.kernels
    ks[0].common_runtime_args = [7]
    pd.kernels = ks
    print("patched", list(pd.kernels[0].common_runtime_args))
    ks = pd.kernels
    ks[0].common_runtime_args[0] = 9
    pd.kernels = ks
    print("patched2", list(pd.kernels[0].common_runtime_args))
    print(
        "dram align",
        ttnn.get_dram_alignment() if hasattr(ttnn, "get_dram_alignment") else None,
        ttnn.get_l1_alignment() if hasattr(ttnn, "get_l1_alignment") else None,
    )
    print(ttnn.Mcast.__init__.__doc__)
finally:
    ttnn.close_device(device)
