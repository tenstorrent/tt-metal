import ttnn

core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
rt[1][0] = [3, 4]
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=core,
    compile_time_args=[7],
    runtime_args=rt,
    common_runtime_args=[100],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
ks = pd.kernels
print("TYPES", type(ks), type(ks[0].common_runtime_args))
pd.kernels[0].common_runtime_args = [555]
print("after set via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args[0] = 777
print("after item set:", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].runtime_args[1][0][0] = 99
print("rt", list(pd.kernels[0].runtime_args[1][0]))
print("UINT16", ttnn.DataType.UINT16, hasattr(ttnn, "uint16"))
print("SWC", ttnn.split_work_to_cores.__doc__)
print("CORES", ttnn.corerange_to_cores.__doc__)
print("CCD", ttnn.ComputeConfigDescriptor.__init__.__doc__)
