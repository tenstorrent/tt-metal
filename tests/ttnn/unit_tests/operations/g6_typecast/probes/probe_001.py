import ttnn

print([a for a in dir(ttnn.ComputeConfigDescriptor) if not a.startswith("_")])
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[1],
    runtime_args=ttnn.RuntimeArgs(),
    common_runtime_args=[5],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
pd.kernels[0].common_runtime_args = [7]
print("after mutate", list(pd.kernels[0].common_runtime_args))
kk = pd.kernels[0]
kk.common_runtime_args = [9]
print("after mutate2", list(pd.kernels[0].common_runtime_args))
r = split = ttnn.split_work_to_cores(ttnn.CoreCoord(13, 10), 17, row_wise=True)
print(r)
print(ttnn.corerange_to_cores(r[2], None, True))
print([a for a in dir(ttnn) if "UnpackToDest" in a])
