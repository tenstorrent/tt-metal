import sys

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from packed_model import BS, SENT, reader_groups

cap = torch.load("/mnt/data/kernel-agent/dev/prefill/runs/c5120-v2/dump/cuts_ev/msa_block_ids.pt")


def rows_for(name):
    if name == "syn":
        gen = torch.Generator().manual_seed(7)
        S = 1280
        q0 = 51200
        q = torch.randn(1, 16, S, 128, generator=gen)
        k = torch.randn(1, 1, 56320, 128, generator=gen)
        v = torch.randn(1, 1, 56320, 128, generator=gen)
        rows = []
        for s in range(S):
            p = q0 + s
            local = p // BS
            pool = torch.randperm(local, generator=gen)[:15]
            rows.append(torch.cat([pool, torch.tensor([local])]).sort().values.tolist())
        return rows, 51200
    L, r = name
    rows = cap[L]["block_ids"][0][r * 1280 : (r + 1) * 1280].tolist()
    return [[(i if 0 <= i < 0xFFFFFFF0 else SENT) for i in x] for x in rows], 51200 + r * 1280


def stats(rows, cs, G, split="ragged"):
    out = []
    for c in range(130):
        st = c * 9 + min(c, 110)
        cnt = 9 + (1 if c < 110 else 0)
        rs = 0
        bl = 0
        gr = 0
        stamps = 0
        hid = 0
        for g in reader_groups(rows, st, cnt, 1280, G, cs):
            gr += 1
            bl += len(g["uid"])
            for e in range(len(g["uid"])):
                for r in range((g["g"] + 1) // 2):
                    m = (g["umask"][e] >> (2 * r)) & 3
                    if m:
                        rs += 1
                        v1 = 2 * r + 1 < g["g"]
                        if m != 3 and v1:
                            hid += 1
        out.append((rs, bl, gr, hid))
    return out


meas = {
    ("l30", 2): 1049.67,
    ("l30", 4): 791.54,
    ("l30", 8): 705.99,
    ("l3", 8): 486.03,
    ("syn", 8): 1421.01,
    ("l30", 1): 1522.82,
    ("l3", 1): 1523.29,
    ("syn", 1): 1506.93,
}
names = {"l30": (30, 0), "l3": (3, 3), "syn": "syn"}
for (n, G), t in meas.items():
    rows, cs = rows_for(names[n])
    if G == 1:
        s = [(16 * (9 + (1 if c < 110 else 0)), 16 * (9 + (1 if c < 110 else 0)), 0, 0) for c in range(130)]
    else:
        s = stats(rows, cs, G)
    mx = max(s, key=lambda x: x[0])
    tot_bl = sum(x[1] for x in s)
    tot_rs = sum(x[0] for x in s)
    print(
        f"{n} G={G}: {t:7.1f} us | max-core rowsteps {mx[0]} blocks {mx[1]} hidden-half {mx[3]} | total blocks {tot_bl} rowsteps {tot_rs} | us/maxrowstep {t/mx[0]:.2f} | GB/s {tot_bl*34816/t/1e3:.0f}"
    )
print("---- larger groups (projected) ----")
for n in ("l30", "l3", "syn"):
    rows, cs = rows_for(names[n])
    for G in (8, 10, 12, 16):
        s = stats(rows, cs, G)
        mx = max(s, key=lambda x: x[0])
        tot_bl = sum(x[1] for x in s)
        print(
            f"{n} G={G}: max-core rowsteps {mx[0]} max-core blocks {max(x[1] for x in s)} total blocks {tot_bl} ({tot_bl/20480:.3f}) DRAM floor@470GB/s {tot_bl*34816/470e3:.0f} us groups/core max {max(x[2] for x in s)}"
        )
