import ttnn

cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
rd = ttnn.KernelDescriptor(
    kernel_source="r.cpp", core_ranges=cr, common_runtime_args=[11], config=ttnn.ReaderConfigDescriptor()
)
wr = ttnn.KernelDescriptor(
    kernel_source="w.cpp", core_ranges=cr, common_runtime_args=[22], config=ttnn.WriterConfigDescriptor()
)
desc = ttnn.ProgramDescriptor(kernels=[rd, wr], semaphores=[], cbs=[])
ks = desc.kernels
print("type kernels:", type(ks), "type crta:", type(ks[0].common_runtime_args))
ks[0].common_runtime_args[0] = 111
print("A elem-assign via held list:", [list(k.common_runtime_args) for k in desc.kernels])
desc.kernels[1].common_runtime_args[0] = 222
print("B elem-assign via fresh access:", [list(k.common_runtime_args) for k in desc.kernels])
print("grid_to_cores:", hasattr(ttnn, "grid_to_cores"), "corerange_to_cores:", hasattr(ttnn, "corerange_to_cores"))
print(ttnn.grid_to_cores(5, 2, 3, True))
