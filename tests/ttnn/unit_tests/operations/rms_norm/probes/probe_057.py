import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:

    def cfg():
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        return c

    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    W = 64
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    mem = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    ss = mem.shard_spec
    print("shard", int(ss.shape[0]), int(ss.shape[1]), "ncores", ss.grid.num_cores())
    xi = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    gi = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(xi, gamma=gi, compute_kernel_config=cfg(), memory_config=xi.memory_config())
    act = ttnn.to_torch(out).reshape(shape).float()[0, 0]
    xf = x.float()[0, 0]
    gf = g.float()
    rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + 1e-6)
    exp = xf / rms * gf
    err = (act - exp).abs().mean(0)
    print("per-col mean-abs err[:16]:", [round(float(e), 2) for e in err[:16]])
    nog = (xf / rms)[0]
    print("row0 act[:8] ", [round(float(v), 2) for v in act[0][:8]])
    print("row0 exp[:8] ", [round(float(v), 2) for v in exp[0][:8]])
    print("row0 x/rms[:8]", [round(float(v), 2) for v in nog[:8]])
    print("row0 gamma[:8]", [round(float(v), 2) for v in gf[:8]])
    print("act/(x/rms)[:8]", [round(float(a / b), 2) if abs(b) > 1e-2 else None for a, b in zip(act[0][:8], nog[:8])])
finally:
    ttnn.close_device(device)
