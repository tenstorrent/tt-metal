import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def ref(x):
    return x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)


cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
for name, mc_fn in [
    ("res", lambda: shard_config([64, 512], (7, 1), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)),
    ("dram", lambda: ttnn.DRAM_MEMORY_CONFIG),
]:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 416, 512)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc_fn())
    o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    err = (o - ref(x)).abs()[0, 0]
    rowerr = err.max(-1).values
    print(
        "R",
        name,
        "max",
        err.max().item(),
        "worst rows",
        torch.topk(rowerr, 5).indices.tolist(),
        "col err max per 32-col block",
        [round(v, 4) for v in err.reshape(416, 16, 32).amax((0, 2)).tolist()],
    )
ttnn.close_device(device)
