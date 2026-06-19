import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)


def regime(shape):
    W = shape[-1]
    x = torch.randn(*shape)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ti.dtype, ti.layout, dev, ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    prog, _ = desc.create_program_descriptor(ti, out, g, 1e-6, cfg)
    return len(prog.semaphores)


try:
    for s in [(1, 24, 4096), (1, 128, 4096)]:
        nsem = regime(s)
        print("RES %s sems=%d -> %s" % (s, nsem, "RegimeB" if nsem == 2 else "RegimeA"))
finally:
    ttnn.close_device(dev)
