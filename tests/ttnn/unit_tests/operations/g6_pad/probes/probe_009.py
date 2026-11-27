import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 1, 8, 62080).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        y = ttnn.reshape(x, ttnn.Shape([1, 1, 16, 62080]), x.padded_shape)
    finally:
        g = ttnn.graph.end_graph_capture()
    progs = [
        n["params"].get("name", "?")
        for n in g
        if n.get("node_type") == "function_start" and "program_factory_type" in n.get("params", {})
    ]
    print("programs:", progs)
    print("fn nodes:", [n["params"].get("name") for n in g if n.get("node_type") == "function_start"])
    print("addr eq", y.buffer_address() == x.buffer_address(), list(y.shape), list(y.padded_shape), list(x.shape))
    # descriptor mutability
    kd = ttnn.KernelDescriptor(
        kernel_source="foo.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        common_runtime_args=[1, 2, 3],
        config=ttnn.WriterConfigDescriptor(),
    )
    kd.common_runtime_args[0] = 99
    print("crt after item set", list(kd.common_runtime_args))
    kd.common_runtime_args = [7, 8, 9]
    print("crt after assign", list(kd.common_runtime_args))
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
    pd.kernels[0].common_runtime_args = [4, 5, 6]
    print("pd kernel crt after assign via pd.kernels[0]", list(pd.kernels[0].common_runtime_args))
    print(
        "num_cores_to_corerangeset",
        ttnn.num_cores_to_corerangeset.__doc__[:400] if ttnn.num_cores_to_corerangeset.__doc__ else None,
    )
    print(device.compute_with_storage_grid_size())
    print(ttnn.num_cores_to_corerangeset(13, device.compute_with_storage_grid_size(), True))
finally:
    ttnn.close_device(device)
