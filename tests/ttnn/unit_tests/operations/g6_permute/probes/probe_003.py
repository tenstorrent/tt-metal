import ttnn

core = ttnn.CoreCoord(0, 0)
crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1],
    runtime_args=rt,
    common_runtime_args=[10, 20],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
print(type(pd.kernels), type(k.common_runtime_args))
ks = pd.kernels
ks[0].common_runtime_args[0] = 99
print("after in-place via list:", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args[1] = 77
print("after direct:", list(pd.kernels[0].common_runtime_args))
pd.custom_program_hash = 12345
print(pd.custom_program_hash)
device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y)
ttnn.close_device(device)
