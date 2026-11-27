import ttnn

crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=crs,
    compile_time_args=[1, 2],
    common_runtime_args=[5],
    config=ttnn.WriterConfigDescriptor(),
)
cb = ttnn.CBDescriptor(
    total_size=2048,
    core_ranges=crs,
    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
print(type(pd.kernels), type(k.common_runtime_args))
pd.kernels[0].common_runtime_args[0] = 99
print("after in-place idx:", list(pd.kernels[0].common_runtime_args))
kk = pd.kernels[0]
kk.common_runtime_args = [77]
print("after attr set on elem:", list(pd.kernels[0].common_runtime_args))
k.common_runtime_args = [55]
print("orig k changed, pd sees:", list(pd.kernels[0].common_runtime_args))
pd.custom_program_hash = (1 << 63) + 12345
print(pd.custom_program_hash)
print(
    ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
        ]
    ).ranges()
)
