# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""When does the level-1 combine (sparse_combine_rf) become link bound against the flat routed expert? CPU model, no
device. Per chip, flat = max(active experts x w0 + a x rows, c x rows) (weight stream vs compute; within 1-4% of the
2x4 harness on uniform / spread, c fitted on the packed run so it includes the hot-expert slowdown). Combine = the
busiest (chip, direction, link)'s rows x 2H bytes at 25 GB/s, divided by the efficiency measured on 2x4 per routing.
Routings as scratch_flat_combine_rf.py (uniform, Zipf 1.2 spread / packed), over the models' real expert counts.
Assumes 2 links per direction and the 2x4 efficiencies on every stage. Run: python <this file>."""

import collections

import torch

MODELS = {
    "K3": dict(
        E=896,
        K=16,
        H=3584,
        w0=37.1,
        a=0.062,
        c=0.180,
        eff={"uniform": 0.72, "zipf_spread_1.2": 0.75, "zipf_packed_1.2": 0.89},
    ),
    "V4.1": dict(
        E=384,
        K=6,
        H=5120,
        w0=39.4,
        a=0.089,
        c=0.247,
        eff={"uniform": 0.64, "zipf_spread_1.2": 0.70, "zipf_packed_1.2": 0.87},
    ),
}
MESH = {"2x4": (2, 4, False, False), "4x4": (4, 4, True, False), "8x4": (8, 4, True, True)}  # R, C, wrap X, wrap Y
CHUNKS = [2560, 5120, 8192, 10240, 16384]  # global tokens per stage
ROUTINGS = ["uniform", "zipf_spread_1.2", "zipf_packed_1.2"]
L, GBPS = 2, 25.0


def route(src, home, R, C, wrap_x, wrap_y):  # (as sparse_combine_l1.route: Y first, then X)
    hops = []
    r, c = src

    def step(n, a, b, wrap, tie_plus):
        if wrap:
            f = (b - a) % n
            plus = f * 2 < n or (f * 2 == n and tie_plus)
            return (1 if plus else -1), (f if plus else n - f)
        return (1 if b > a else -1), abs(b - a)

    s, k = step(R, r, home[0], wrap_y, False)
    for _ in range(k):
        hops.append(((r, c), "Y+" if s > 0 else "Y-"))
        r = (r + s) % R
    s, k = step(C, c, home[1], wrap_x, True)
    for _ in range(k):
        hops.append(((r, c), "X+" if s > 0 else "X-"))
        c = (c + s) % C
    return hops


def ids(name, N, S, E, K, g):  # (as scratch_flat_combine_rf._ids)
    EPC = E // N
    if name == "uniform":
        return torch.stack([torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(S)]) for _ in range(N)])
    s = float(name.split("_")[-1])
    r = torch.arange(E)
    perm = r if "packed" in name else (r % N) * EPC + r // N
    w = torch.zeros(E)
    w[perm] = 1.0 / (r.double() + 1).pow(s).float()
    return torch.multinomial(w.expand(N * S, E), K, replacement=False, generator=g).view(N, S, K)


def model(p, R, C, wx, wy, T, rname):
    """(flat us on the slowest chip, combine us now)."""
    N = R * C
    E, K = p["E"], p["K"]
    EPC, S = E // N, T // N
    x = ids(rname, N, S, E, K, torch.Generator().manual_seed(0))
    cnt = torch.bincount(x.flatten(), minlength=E).view(N, EPC)
    flat = max(
        max((cnt[ch] > 0).sum().item() * p["w0"] + p["a"] * cnt[ch].sum().item(), p["c"] * cnt[ch].sum().item())
        for ch in range(N)
    )
    src = x // EPC
    lk = (torch.arange(S).view(1, S, 1) + torch.arange(K).view(1, 1, K)) % L
    home = torch.arange(N).view(N, 1, 1).expand_as(src)
    m = torch.bincount(((src * N + home) * L + lk).flatten(), minlength=N * N * L).view(N, N, L)
    chips = [(r, c) for r in range(R) for c in range(C)]
    load = collections.Counter()
    for s in range(N):
        for h in range(N):
            for hop in route(chips[s], chips[h], R, C, wx, wy) if s != h else []:
                for l in range(L):
                    load[hop + (l,)] += m[s, h, l].item()
    return flat, max(load.values()) * p["H"] * 2 / (GBPS * 1e3) / p["eff"][rname]


if __name__ == "__main__":
    for rname in ROUTINGS:
        print(f"\n{rname}: combine now / flat (us), ratio (* = link bound)")
        for mn, p in MODELS.items():
            for st, (R, C, wx, wy) in MESH.items():
                cells = []
                for T in CHUNKS:
                    f, cb = model(p, R, C, wx, wy, T, rname)
                    cells.append(f"{T / 1024:g}K {cb:6.0f}/{f:6.0f} {cb / f:4.2f}{'*' if cb > f else ' '}")
                print(f"{mn:5s} {st}  " + " | ".join(cells))
