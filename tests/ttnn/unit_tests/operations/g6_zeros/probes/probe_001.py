import ttnn

d = ttnn.open_device(device_id=0)
print("type", type(d), ttnn.Device is ttnn.MeshDevice, isinstance(d, ttnn.MeshDevice))
print("grid", d.compute_with_storage_grid_size(), "dram", d.dram_grid_size(), "arch", d.arch(), type(d.arch()))
print("shape", d.shape, "id", d.id() if hasattr(d, "id") else None)
print("num_dram_channels", getattr(d, "num_dram_channels", lambda: None)())
t = ttnn.allocate_tensor_on_device(
    ttnn.Shape([8, 12, 128, 128]), ttnn.float32, ttnn.TILE_LAYOUT, d, ttnn.DRAM_MEMORY_CONFIG
)
print("pages", t.buffer_num_pages(), t.buffer_page_size(), t.buffer_aligned_page_size(), hex(t.buffer_address()))
for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
    u = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 256, 128]), dt, ttnn.TILE_LAYOUT, d, ttnn.DRAM_MEMORY_CONFIG)
    print(dt, u.buffer_num_pages(), u.buffer_page_size(), u.buffer_aligned_page_size())
core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
rt[1][0] = [3, 4]
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=core,
    compile_time_args=[1],
    runtime_args=rt,
    common_runtime_args=[7, 8],
    config=ttnn.WriterConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
pd.kernels[0].common_runtime_args = [9, 10]
print("mut via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args))
kk = pd.kernels[0]
kk.common_runtime_args[0] = 11
print("mut via elem:", list(pd.kernels[0].common_runtime_args))
print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
print(type(ttnn.MeshDevice), [a for a in dir(d) if "id" in a.lower()][:20])
ttnn.close_device(d)
