# CPU: how concentrated is the true top-2048 over columns / local blocks (decides k_loc, k_blk). fp32 scores, 64-token-slice layout over 8 columns.
import json

import torch

torch.set_num_threads(8)
HEADS, DIM, K = 32, 128, 2048


def run(N, mode, rows=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(N, DIM, generator=g)
    if mode == "heavy":
        G = G * torch.exp(torch.randn(N, 1, generator=g))
    q = torch.randn(HEADS, rows, DIM, generator=g)
    w = torch.rand(rows, HEADS, generator=g) * 0.2
    if mode == "heavy":
        for s in range(rows):
            hot = torch.randint(0, N, (400,), generator=g)
            G[hot] += 6 * q[:, s].mean(0, keepdim=True)
        G[N - 700 :] += 3.2 * q[:, 0].mean(0, keepdim=True)  # recent window block
    if mode == "recent":  # recency-dominated head: the last 2048 tokens all outscore everything
        G[N - 2048 :] += 8 * q[:, 0].mean(0, keepdim=True)
    sc = torch.zeros(rows, N)
    for s in range(rows):
        for c0 in range(0, N, 131072):
            sc[s, c0 : c0 + 131072] = (torch.relu(q[:, s] @ G[c0 : c0 + 131072].T) * w[s][:, None]).sum(0)
    out = {"N": N, "mode": mode}
    n = N // 8
    for B in (16, 32):
        mc = mb = 0
        cols = []
        for s in range(rows):
            top = torch.topk(sc[s], K).indices
            col = (top // 64) % 8
            loc = (top // 512) * 64 + (top % 64)
            cnt = torch.bincount(col, minlength=8)
            cols.append(int(cnt.max()))
            blk = loc // (n // B)
            cb = torch.bincount(col * B + blk, minlength=8 * B)
            mb = max(mb, int(cb.max()))
        out[f"max_col_share"] = max(cols)
        out[f"max_blk_share_B{B}"] = mb
    return out


for N in (131072 * 2, 1048576):
    for mode in ("random", "heavy", "recent"):
        print(json.dumps(run(N, mode)), flush=True)
