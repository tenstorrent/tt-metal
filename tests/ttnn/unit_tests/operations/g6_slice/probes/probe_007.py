import ttnn

k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),
    compile_time_args=[1, 2],
    runtime_args=[(ttnn.CoreCoord(0, 0), [5, 6]), (ttnn.CoreCoord(1, 0), [7, 8])],
    common_runtime_args=[11, 12],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
pd.kernels[0].common_runtime_args[0] = 99
pd.kernels[0].runtime_args[1][0][1] = 77
print(
    list(pd.kernels[0].common_runtime_args),
    list(pd.kernels[0].runtime_args[1][0]),
    list(pd.kernels[0].runtime_args[0][0]),
)
print(type(pd.kernels), type(pd.kernels[0].common_runtime_args))
print(
    hasattr(ttnn, "split_work_to_cores"),
    hasattr(ttnn.device, "get_max_worker_l1_unreserved_size"),
    hasattr(ttnn, "corerange_to_cores"),
)
