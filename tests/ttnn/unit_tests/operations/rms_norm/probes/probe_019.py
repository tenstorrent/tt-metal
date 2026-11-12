import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 512)
mc = shard_config(
    [64, 512], (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32
)
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()[0, 0]
x = x[0, 0]
r = x / torch.sqrt(torch.mean(x * x, -1, keepdim=True) + 1e-6)
e = (o - r).abs()
print("R per tile err", [[round(v, 3) for v in row] for row in e.reshape(2, 32, 16, 32).amax((1, 3)).tolist()])
implied = o / x
print(
    "R row33 implied rstd per tile",
    [round(v, 4) for v in implied[33].reshape(16, 32).median(-1).values.tolist()],
    "true",
    (1 / torch.sqrt(torch.mean(x[33] ** 2) + 1e-6)).item(),
)
print(
    "R row1 implied rstd per tile",
    [round(v, 4) for v in implied[1].reshape(16, 32).median(-1).values.tolist()],
    "true",
    (1 / torch.sqrt(torch.mean(x[1] ** 2) + 1e-6)).item(),
)
imp = implied.median(-1).values
tru = 1 / torch.sqrt(torch.mean(x * x, -1) + 1e-6)
ss_imp = 512 / imp**2
ss = (x * x).sum(-1)
for rr in range(32, 40):
    d = (tru - imp[rr]).abs()
    j = int(d.argmin())
    print(
        "R row",
        rr,
        "imp",
        round(imp[rr].item(), 4),
        "nearest true row",
        j,
        round(d[j].item(), 5),
        "ss_imp",
        round(ss_imp[rr].item(), 2),
        "ss",
        round(ss[rr].item(), 2),
    )
# hypothesis: ss_imp(tr1 row i) = ss(row i) + ss(row?) - ...
print("R diff ss_imp-ss tr1", [round(v, 2) for v in (ss_imp[32:48] - ss[32:48]).tolist()])
print("R ss tr0", [round(v, 2) for v in ss[0:16].tolist()])
ttnn.close_device(device)
