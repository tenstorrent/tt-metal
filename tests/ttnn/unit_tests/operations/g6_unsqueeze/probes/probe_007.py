import ttnn

k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),
    compile_time_args=[1],
    common_runtime_args=[5, 6],
    config=ttnn.ReaderConfigDescriptor(),
)
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
rt[1][0] = [3, 4]
k.runtime_args = rt
d = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
print(type(d.kernels), type(k.common_runtime_args))
d.kernels[0].common_runtime_args[0] = 99
print("after mod via d.kernels:", list(d.kernels[0].common_runtime_args))
k.common_runtime_args[0] = 77
print("k:", list(k.common_runtime_args), "d:", list(d.kernels[0].common_runtime_args))
kl = d.kernels
kl[0].common_runtime_args[1] = 55
print("kl:", list(kl[0].common_runtime_args), "d:", list(d.kernels[0].common_runtime_args))
print([x for x in dir(ttnn.DataMovementConfigDescriptor) if not x.startswith("_")])
print([x for x in dir(ttnn.DataMovementProcessor) if not x.startswith("_")])
