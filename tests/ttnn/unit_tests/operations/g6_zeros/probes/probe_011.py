import ttnn

crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2, 3]
rt[1][0] = [4, 5, 6]
kd = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1],
    runtime_args=rt,
    common_runtime_args=[0],
    config=ttnn.WriterConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
print(type(pd.kernels), type(pd.kernels[0].common_runtime_args))
print("same list obj twice?", pd.kernels is pd.kernels)
pd.kernels[0].common_runtime_args[0] = 77
print("after elem-mutate via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args))
k0 = pd.kernels[0]
k0.common_runtime_args[0] = 88
print("after mutate via held k0:", list(pd.kernels[0].common_runtime_args))
k0.common_runtime_args = [99]
print("after assign via held k0:", list(pd.kernels[0].common_runtime_args))
