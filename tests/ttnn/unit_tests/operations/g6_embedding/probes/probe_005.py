import ttnn

core = ttnn.CoreCoord(0, 0)
crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, ttnn.CoreCoord(1, 0))])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2, 3]
rt[1][0] = [4, 5, 6]
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[],
    runtime_args=rt,
    common_runtime_args=[7, 8],
    config=ttnn.ReaderConfigDescriptor(),
)
desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
print("type kernels", type(desc.kernels))
kd = desc.kernels[0]
kd.common_runtime_args = [70, 80]
print("after assign via desc.kernels[0]:", list(desc.kernels[0].common_runtime_args))
c = desc.kernels[0].common_runtime_args
print("type cra", type(c))
c[0] = 700
print("after item set:", list(desc.kernels[0].common_runtime_args))
r = desc.kernels[0].runtime_args[1][0]
print("type rt", type(r))
r[0] = 400
print("rt after item set:", list(desc.kernels[0].runtime_args[1][0]))
