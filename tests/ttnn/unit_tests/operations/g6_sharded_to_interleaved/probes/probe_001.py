# Probe: nanobind mutation semantics of ProgramDescriptor and shard->core order.
import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
    spec = ttnn.ShardSpec(grid, (32, 160), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    a = ttnn.from_torch(
        torch.randn(1, 5120).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    b = ttnn.from_torch(
        torch.randn(1, 5120).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    print("a addr", hex(a.buffer_address()), "b addr", hex(b.buffer_address()))
    print(
        "padded",
        a.padded_shape,
        "aligned page",
        a.buffer_aligned_page_size(),
        "page",
        a.buffer_page_size(),
        "tile_size",
        ttnn.tile_size(ttnn.bfloat16),
        ttnn.tile_size(ttnn.bfloat8_b),
    )
    s = a.memory_config().shard_spec
    print("shard shape", s.shape, "orient", s.orientation, "grid", s.grid, type(s.grid))
    cores = ttnn.corerange_to_cores(s.grid, None, row_wise=True)
    print("row_wise first 10", [(c.x, c.y) for c in cores[:10]])
    cores = ttnn.corerange_to_cores(s.grid, None, row_wise=False)
    print("col first 10", [(c.x, c.y) for c in cores[:10]])
    cb = ttnn.cb_descriptor_from_sharded_tensor(0, a)
    print("cb total", cb.total_size, "addr", hex(cb.buffer_address()), "ranges", cb.core_ranges)
    kd = ttnn.KernelDescriptor(
        kernel_source="x.cpp",
        core_ranges=grid,
        compile_time_args=[1],
        runtime_args=[],
        common_runtime_args=[7],
        config=ttnn.WriterConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])
    print("type common", type(pd.kernels[0].common_runtime_args))
    pd.kernels[0].common_runtime_args[0] = 99
    print("after inplace set via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args))
    k = pd.kernels[0]
    k.common_runtime_args = [123]
    print("after assign via k:", list(pd.kernels[0].common_runtime_args))
    cbb = ttnn.cb_descriptor_from_sharded_tensor(0, b)
    pd.cbs[0].set_buffer_from_cb(cbb)
    print("after set_buffer_from_cb: pd.cbs[0] addr", hex(pd.cbs[0].buffer_address()))
    print("L1 banks", device.num_banks(ttnn.BufferType.L1) if hasattr(device, "num_banks") else "n/a")
    print("dram banks", device.num_banks(ttnn.BufferType.DRAM) if hasattr(device, "num_banks") else "n/a")
    print("grid size", device.compute_with_storage_grid_size())
    print("device id", device.id(), type(device))
finally:
    ttnn.close_device(device)
