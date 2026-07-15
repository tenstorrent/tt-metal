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
    xi = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    gi = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(xi, gamma=gi, compute_kernel_config=cfg(), memory_config=xi.memory_config())
    _ = ttnn.to_torch(out)
    print("done")
finally:
    ttnn.close_device(device)
