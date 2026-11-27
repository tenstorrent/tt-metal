import ttnn

print(ttnn.split_work_to_cores.__doc__[:900])
print([n for n in dir(ttnn._ttnn.reports) if not n.startswith("_")])
print([n for n in dir(ttnn.TensorAccessorArgs) if not n.startswith("_")])
print(
    [n for n in dir(ttnn.DataMovementProcessor) if not n.startswith("_")],
    [n for n in dir(ttnn.NOC) if not n.startswith("_")],
)
kd = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[1],
    runtime_args=ttnn.RuntimeArgs(),
    common_runtime_args=[5, 6],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
k = pd.kernels[0]
print(type(pd.kernels), type(k.common_runtime_args))
k.common_runtime_args[0] = 99
print(list(pd.kernels[0].common_runtime_args))
pd.custom_program_hash = 12345
print(pd.custom_program_hash)
grid = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),
    ]
)
print([(r.start.x, r.start.y) for r in grid.ranges()])
print([(c.x, c.y) for c in ttnn.corerange_to_cores(grid, row_wise=True)])
