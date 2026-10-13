"""t479 (copy of t456/replay456.py, + mass kept by the LightX2V selection): CPU replay of one SLA call dumped by plug/t479_dump.py (device 0 = SP shard 0, TP group 0).

(a) kernel: device vsa_sdpa output vs CPU block-sparse attention over the device's own ids;
(b) selection: device ids vs LightX2V SLA selection (q pooled over 128 rows, smoothed k over 64, top int(0.15*K));
(c) sparsity: device output vs dense attention over all real keys, and the softmax mass the selection keeps.
Usage: replay479.py <call<i>.pt> [seq_len=75449] [sparsity=0.85]
"""
import math
import sys

import torch

d = torch.load(sys.argv[1])
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 75449
sparsity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
T = 64
print("call", d.get("call"), "shapes", d["shapes"], "kwargs", d["kwargs"], "skipped", d["skipped_calls"])
print("counts", d["counts"].shape, d["counts"].flatten()[:8].tolist(), d["counts"].flatten()[-8:].tolist())
q, k, v, out = (d[n][0].float() for n in ("q", "k", "v", "out"))  # [H, rows, D]
raw = d["ids"][0].to(torch.int64)  # [H, local tiles, k_pad]
H, rows, D = q.shape
shift = d["kwargs"]["coarse_slots_shift"]
tiles = d["kwargs"]["coarse_real_per_shard"]
kk = d["kwargs"]["list_len"]
ids = (raw - (raw >> shift) * ((1 << shift) - tiles))[..., :kk]
n_real = -(-seq_len // T)
cnt = torch.full((k.shape[1] // T,), T)
cnt[n_real - 1] = seq_len - (n_real - 1) * T
cnt[n_real:] = 0
print(f"H={H} local rows={rows} k_rows={k.shape[1]} list_len={kk} shift={shift} tiles/shard={tiles} n_real={n_real}")
print(
    "ids range",
    int(ids.min()),
    int(ids.max()),
    "dups",
    sum(len(set(ids[h, t].tolist())) < kk for h in range(H) for t in range(ids.shape[1])),
)
print("abs max q/k/v/out", [round(float(x.abs().max()), 2) for x in (q, k, v, out)])

pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
keys_of = lambda blocks: torch.cat([torch.arange(b * T, b * T + int(cnt[b])) for b in sorted(blocks)])
real = keys_of(range(n_real))
sm = 1 / math.sqrt(D)

# (b) LightX2V selection for this shard's q blocks over all real keys
kr = k[:, real]
kpool = (
    torch.nn.functional.pad(kr - kr.mean(1, keepdim=True), (0, 0, 0, n_real * T - seq_len))
    .reshape(H, n_real, T, D)
    .sum(2)
    / cnt[:n_real, None].float()
)
qpool = q.reshape(H, rows // 128, 128, D).mean(2)
ref = torch.topk(qpool @ kpool.transpose(-1, -2), int((1 - sparsity) * n_real), dim=-1).indices
ov = torch.tensor(
    [[len(set(ids[h, t].tolist()) & set(ref[h, t // 2].tolist())) / kk for t in range(ids.shape[1])] for h in range(H)]
)
print(f"(b) overlap with LightX2V: mean {ov.mean():.4f} min {ov.min():.4f}; ref k {ref.shape[-1]} vs device {kk}")

# (a)/(c) per sampled q tile
g = torch.Generator().manual_seed(0)
sample = sorted(set(torch.randperm(ids.shape[1], generator=g)[:24].tolist() + [0, 7, 8, 39, 40, 46, 47, 100]))
rows_a, rows_c, rows_r, mass, mass_r = [], [], [], [], []
for h in range(H):
    for t in sample:
        r = slice(t * T, (t + 1) * T)
        keys = keys_of(ids[h, t].tolist())
        s_all = (q[h, r] @ k[h, real].T) * sm
        p_all = torch.softmax(s_all, -1)
        dense = p_all @ v[h, real]
        sparse = torch.softmax((q[h, r] @ k[h, keys].T) * sm, -1) @ v[h, keys]
        keys_ref = keys_of(ref[h, t // 2].tolist())
        sparse_ref = torch.softmax((q[h, r] @ k[h, keys_ref].T) * sm, -1) @ v[h, keys_ref]
        sel = torch.zeros(len(real), dtype=torch.bool)
        sel[torch.searchsorted(real, keys)] = True
        rows_a.append((h, t, pcc(out[h, r], sparse)))
        rows_c.append((h, t, pcc(out[h, r], dense)))
        rows_r.append((h, t, pcc(sparse_ref, dense)))
        mass.append((h, t, float(p_all[:, sel].sum(-1).mean())))
        sel_r = torch.zeros(len(real), dtype=torch.bool)
        sel_r[torch.searchsorted(real, keys_ref)] = True
        mass_r.append((h, t, float(p_all[:, sel_r].sum(-1).mean())))
for name, rs in (
    ("(a) device out vs CPU sparse(device ids)", rows_a),
    ("(c) device out vs dense", rows_c),
    ("(c') LightX2V sparse vs dense", rows_r),
    ("(c) softmax mass kept", mass),
    ("(c') softmax mass kept by LightX2V ids", mass_r),
):
    vals = torch.tensor([x[2] for x in rs])
    worst = sorted(rs, key=lambda x: x[2])[:4]
    print(
        f"{name}: mean {vals.mean():.4f} min {vals.min():.4f} worst (h,t,val) {[(a, b, round(c, 4)) for a, b, c in worst]}"
    )
