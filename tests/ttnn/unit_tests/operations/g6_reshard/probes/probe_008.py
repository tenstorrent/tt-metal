import ttnn

crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
kd = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1],
    runtime_args=rt,
    common_runtime_args=[10, 20],
    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
)
print("T1", type(kd.common_runtime_args), list(kd.common_runtime_args))
kd.common_runtime_args[0] = 99
print("T2 after inplace on kd:", list(kd.common_runtime_args))
pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
k0 = pd.kernels[0]
k0.common_runtime_args[1] = 77
print("T3 pd after mutate via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args), type(pd.kernels))
