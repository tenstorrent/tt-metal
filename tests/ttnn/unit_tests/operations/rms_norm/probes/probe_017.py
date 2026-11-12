import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi4
cfg.fp32_dest_acc_en = True
shape = (1, 1, 32 * 240, 512)
torch.manual_seed(0)
x = torch.randn(shape)
tx = ttnn.from_torch(
    x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
o = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()[0, 0]
x = x[0, 0]
implied = (o / x).median(-1).values
imp_ss = 512 / implied[32:64] ** 2
# per-tile sumsq: tiles[tr, tc, row] (row within tile)
t = (x * x).reshape(240, 32, 16, 32).sum(-1)  # [tr, row, tc]
seq = torch.cat([t[0], t[1], t[2]], dim=-1)  # [row, 48 tiles] flattened tile-major for tr0..2
best = []
for a in range(0, 33):
    cand = seq[:, a : a + 16].sum(-1)
    best.append(((cand - imp_ss).abs().mean().item(), a))
best.sort()
print("R best offsets", best[:3])
# also 2x combos
cand = seq[:, 16:32].sum(-1) * 2 - seq[:, 0:16].sum(-1)
print("R 2*tr1 - tr0 err", (cand - imp_ss).abs().mean().item())
cand = seq[:, 16:32].sum(-1) + seq[:, 0:16].sum(-1)
print("R tr1+tr0 err", (cand - imp_ss).abs().mean().item())
cand = seq[:, 16:32].sum(-1) + seq[:, 0:1].sum(-1)
print("R tr1+tile0 err", (cand - imp_ss).abs().mean().item())
for j in range(16):
    cand = seq[:, 16:32].sum(-1) - seq[:, 16 + j]
    e = (cand - imp_ss).abs().mean().item()
    if e < 3:
        print("R tr1 minus tile", j, e)
for j in range(48):
    cand = seq[:, 16:32].sum(-1) + seq[:, j]
    e = (cand - imp_ss).abs().mean().item()
    if e < 3:
        print("R tr1 plus tile", j, e)
cand = seq[:, 17:32].sum(-1)
print("R tr1 minus first", (cand - imp_ss).abs().mean().item())
ttnn.close_device(device)
