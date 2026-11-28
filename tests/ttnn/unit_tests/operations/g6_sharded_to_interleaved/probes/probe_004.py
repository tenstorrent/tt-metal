import ttnn

crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
kd = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1],
    runtime_args=rt,
    common_runtime_args=[7],
    config=ttnn.WriterConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
k0 = pd.kernels[0]
k0.common_runtime_args = [9]
print("via handle", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args[0] = 11
print("via index", list(pd.kernels[0].common_runtime_args))
print("kd orig", list(kd.common_runtime_args))
print(type(pd.kernels))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1)) if False else None
