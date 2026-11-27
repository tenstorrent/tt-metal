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
r = ttnn.split_work_to_cores(ttnn.CoreCoord(11, 10), 32, row_wise=True)
print("split", r[0], r[1], r[2], r[3], r[4], r[5])
print("split", [(c.x, c.y) for c in ttnn.corerange_to_cores(r[2], None, True)][:14])
