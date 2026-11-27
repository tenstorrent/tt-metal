# No device: checks whether ProgramDescriptor.kernels elements and common_runtime_args mutate in place.
import ttnn

crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1, 2],
    common_runtime_args=[10, 20, 30],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
ks = pd.kernels
print(type(ks), type(ks[0]), type(ks[0].common_runtime_args))
ks[0].common_runtime_args[0] = 99
print("after elem set via list elem:", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args[1] = 77
print("after direct set:", list(pd.kernels[0].common_runtime_args))
print("orig k:", list(k.common_runtime_args))
