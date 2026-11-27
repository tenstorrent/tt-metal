import ttnn

k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[1],
    common_runtime_args=[5],
    runtime_args=ttnn.RuntimeArgs(),
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
pd.kernels[0].common_runtime_args = [7]
print("via pd.kernels:", list(pd.kernels[0].common_runtime_args))
k.common_runtime_args = [9]
print("via k (pd):", list(pd.kernels[0].common_runtime_args), "k:", list(k.common_runtime_args))
print(type(pd.kernels))
r = ttnn.split_work_to_cores(ttnn.CoreCoord(11, 10), 32, row_wise=True)
print(r)
print(ttnn.corerange_to_cores(r[1], None, True)[:14])
