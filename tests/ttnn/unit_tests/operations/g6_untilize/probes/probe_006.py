import ttnn

cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
kd = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=cr,
    compile_time_args=[1],
    common_runtime_args=[5],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
print("TYPES", type(pd.kernels), type(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args[0] = 7
print("after item set", list(pd.kernels[0].common_runtime_args))
k = pd.kernels[0]
k.common_runtime_args = [9]
print("after attr set via held", list(pd.kernels[0].common_runtime_args))
kd.common_runtime_args[0] = 11
print("after orig mutate", list(pd.kernels[0].common_runtime_args))
