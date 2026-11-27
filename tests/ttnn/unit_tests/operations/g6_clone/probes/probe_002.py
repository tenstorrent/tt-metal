import torch, ttnn

cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
k = ttnn.KernelDescriptor(
    kernel_source="a.cpp",
    core_ranges=cr,
    compile_time_args=[1],
    common_runtime_args=[5],
    config=ttnn.ReaderConfigDescriptor(),
)
d = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
d.kernels[0].common_runtime_args = [7]
print("after direct mutate:", list(d.kernels[0].common_runtime_args))
from ttnn.operations.g6_clone.g6_clone_program_descriptor import set_buffer_addresses

k2 = ttnn.KernelDescriptor(
    kernel_source="b.cpp", core_ranges=cr, common_runtime_args=[6], config=ttnn.WriterConfigDescriptor()
)
d2 = ttnn.ProgramDescriptor(kernels=[k, k2], semaphores=[], cbs=[])
set_buffer_addresses(d2, 111, 222)
print("after set_buffer_addresses:", [list(x.common_runtime_args) for x in d2.kernels])
device = ttnn.open_device(device_id=0)
from ttnn.operations.g6_clone import g6_clone_plan

for shape in [(1, 1), (1, 12, 128, 128), (8, 12, 128, 128), (16, 12, 128, 128)]:
    x = ttnn.from_torch(
        torch.randn(shape),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p = g6_clone_plan(x)
    print(
        shape,
        {k: v for k, v in p.items() if k not in ("roles",)},
        "nroles",
        len(p["roles"]),
        p["roles"][:2],
        p["roles"][-1],
    )
print("device id", device.id())
ttnn.close_device(device)
