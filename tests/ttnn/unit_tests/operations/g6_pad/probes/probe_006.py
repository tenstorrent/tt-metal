import ttnn

core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=core,
    compile_time_args=[1],
    runtime_args=[],
    common_runtime_args=[5, 6],
    config=ttnn.WriterConfigDescriptor(),
)
d = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
d.kernels[0].common_runtime_args = [7, 8]
print("via desc.kernels[0]:", list(d.kernels[0].common_runtime_args))
ks = d.kernels
ks[0].common_runtime_args = [10, 11]
print("via held list element:", list(d.kernels[0].common_runtime_args))
print(type(d.kernels), type(d.kernels[0].common_runtime_args))
doc = ttnn.reshape.__doc__
for line in doc.splitlines():
    if "reshape(" in line or "padded" in line:
        print(line)
