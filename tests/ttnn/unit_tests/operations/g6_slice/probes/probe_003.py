import ttnn

k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[1],
    common_runtime_args=[7, 8],
    config=ttnn.ReaderConfigDescriptor(),
)
rt = ttnn.RuntimeArgs()
rt[0][0] = [1, 2]
k.runtime_args = rt
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
print("kernels type", type(pd.kernels))
pd.kernels[0].common_runtime_args[0] = 99
print("after mutate via pd.kernels[0]", list(pd.kernels[0].common_runtime_args))
ks = pd.kernels
ks[0].runtime_args[0][0][0] = 55
print("rt after", list(pd.kernels[0].runtime_args[0][0]))
print(hasattr(ttnn.device, "get_max_worker_l1_unreserved_size"))
import inspect

print(ttnn.split_work_to_cores.__doc__[:600])
print([n for n in dir(ttnn) if "corerange_to_cores" in n or "num_cores_to_corerange" in n])
device = ttnn.open_device(device_id=0)
print("grid", device.compute_with_storage_grid_size())
print(
    "l1",
    ttnn.device.get_max_worker_l1_unreserved_size(device)
    if hasattr(ttnn.device, "get_max_worker_l1_unreserved_size")
    else None,
)
r = ttnn.split_work_to_cores(device.compute_with_storage_grid_size(), 1940, row_wise=True)
print(r)
print(device.id())
ttnn.close_device(device)
