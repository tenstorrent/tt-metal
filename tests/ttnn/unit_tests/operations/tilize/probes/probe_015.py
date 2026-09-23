import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
shape = [1, 1, 32, 32]
x = torch.arange(32 * 32).reshape(shape).remainder(256).to(torch.uint8)
t = ttnn.from_torch(
    x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
for name, ckc in [("default", None), ("fp32dest", ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True))]:
    y = tilize(t, ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)
    o = ttnn.to_torch(y)
    print("RES", name, torch.equal(o, x), o.flatten()[:40].tolist())
ttnn.close_device(dev)
