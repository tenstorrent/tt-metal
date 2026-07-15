import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    print(
        "DRAM_ALIGN",
        ttnn._ttnn.device.get_dram_alignment() if hasattr(ttnn._ttnn.device, "get_dram_alignment") else "?",
        "L1_ALIGN",
        ttnn._ttnn.device.get_l1_alignment(),
    )

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
    xi = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    gi = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(xi, gamma=gi, compute_kernel_config=cfg(), memory_config=xi.memory_config())
    act = ttnn.to_torch(out).reshape(shape).float()[0, 0]
    xf = x.float()[0, 0]
    rms = torch.sqrt((xf**2).mean(-1, keepdim=True) + 1e-6)
    exp = xf / rms * g.float()
    err = (act - exp).abs().mean(0)
    # per-core (Ws=8) err
    for c in range(8):
        print(f"err_by_core core{c} cols[{c*8}:{c*8+8}] = {round(float(err[c*8:c*8+8].mean()),3)}")
finally:
    ttnn.close_device(device)
