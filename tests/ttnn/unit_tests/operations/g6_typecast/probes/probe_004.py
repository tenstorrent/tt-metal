import ttnn, torch

print(ttnn.split_work_to_cores.__doc__[:1500])
print([n for n in dir(ttnn) if "corerangeset" in n.lower() or "num_cores" in n.lower()])
d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print([n for n in dir(d) if "bank" in n.lower() or "alloc" in n.lower()])
try:
    print("num_banks L1", d.num_banks(ttnn.BufferType.L1), "DRAM", d.num_banks(ttnn.BufferType.DRAM))
except Exception as e:
    print("num_banks err", e)
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[],
    runtime_args=[],
    common_runtime_args=[1, 2],
    config=ttnn.ReaderConfigDescriptor(),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
print(type(pd.kernels))
kk = pd.kernels[0]
kk.common_runtime_args = [7, 8]
print("after set via pd.kernels[0]:", pd.kernels[0].common_runtime_args)
print([f for f in dir(ttnn.ComputeConfigDescriptor) if not f.startswith("_")])
print(ttnn.num_cores_to_corerangeset.__doc__[:600] if hasattr(ttnn, "num_cores_to_corerangeset") else "no ncc")
ttnn.close_device(d)
