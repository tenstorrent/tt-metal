"""Run under scripts/tt-probe.sh: dumps per-core NoC coords + DRAM bank NoC coords to geometry.json."""
import json, os
import torch, ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
n = len(cores)
out = ttnn.from_torch(
    torch.zeros(n, 32, dtype=torch.int32),
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=d,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
rt = ttnn.RuntimeArgs()
for i, c in enumerate(cores):
    rt[c.x][c.y] = [out.buffer_address(), i]
k = ttnn.KernelDescriptor(
    kernel_source=os.path.join(HERE, "geom_kernel.cpp"),
    core_ranges=crs,
    compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
    runtime_args=rt,
    config=ttnn.ReaderConfigDescriptor(),
)
cb = ttnn.CBDescriptor(
    total_size=128,
    core_ranges=crs,
    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=128)],
)
ttnn.generic_op([out, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
t = ttnn.to_torch(out).to(torch.int64) & 0xFFFFFFFF
res = {"cores": {}, "banks": {}}
dec = lambda hi: [int((hi >> 4) & 0x3F), int((hi >> 10) & 0x3F)]
for i, c in enumerate(cores):
    r = t[i]
    assert int(r[28]) == i
    res["cores"][f"{c.x},{c.y}"] = {
        "noc0": [int(r[0]), int(r[1])],
        "noc1": [int(r[2]), int(r[3])],
        "virt": [d.worker_core_from_logical_core(c).x, d.worker_core_from_logical_core(c).y],
    }
    banks = {"noc0": [dec(int(r[4 + b])) for b in range(12)], "noc1": [dec(int(r[16 + b])) for b in range(12)]}
    if i == 0:
        res["banks"] = banks
    assert banks == res["banks"]
json.dump(res, open(os.path.join(HERE, "geometry.json"), "w"), indent=1)
print(json.dumps(res["banks"]))
for k_, v in list(res["cores"].items())[:10]:
    print(k_, v)
ttnn.close_device(d)
