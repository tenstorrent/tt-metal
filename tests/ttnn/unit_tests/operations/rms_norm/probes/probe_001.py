import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
dev = ttnn.open_device(device_id=0)
def ns():
    ttnn.ReadDeviceProfiler(dev)
    d = ttnn.get_latest_programs_perf_data()
    r = []
    for _c, progs in (d or {}).items():
        for p in progs:
            x = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if x is not None: r.append(int(x.duration))
    return r
for shape in [(1,1,32,7168),(1,1,640,6144),(1,1,8192,7168),(1,1,7168,1024)]:
    x = ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = ttnn.from_torch(torch.randn(1,1,1,shape[-1]).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for acc in (True, False):
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=acc, math_approx_mode=False)
        rms_norm(x, gamma=g, compute_kernel_config=cfg); ttnn.synchronize_device(dev); ns()
        rms_norm(x, gamma=g, compute_kernel_config=cfg); ttnn.synchronize_device(dev)
        print("PERF", shape, "fp32acc" if acc else "bf16acc", ns()[-1])
ttnn.close_device(dev)
