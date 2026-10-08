"""Per-chip routed rows (routing.pt from BENCH_CAPTURE_ROUTING) vs per-chip MoE op device time (tracy ops.csv of the
same config, layers 3-7). Usage: python correlate.py routing.pt ops.csv [--rows 51200:56320] [--placement p.pt]"""
import argparse
import importlib.util
import re

import pandas as pd
import torch

spec = importlib.util.spec_from_file_location(
    "ep", "/mnt/data/kernel-agent/dev/prefill-moe/tt-metal/models/demos/minimax_m3/utils/expert_placement.py"
)
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)
ap = argparse.ArgumentParser()
ap.add_argument("routing")
ap.add_argument("csv")
ap.add_argument("--rows", default="51200:56320")
ap.add_argument("--placement")
a = ap.parse_args()
cap = torch.load(a.routing, weights_only=False)
dev_ids = cap["device_ids"]
R, C = cap["mesh"]
lo, hi = (int(x) for x in a.rows.split(":"))
plc = ep.load(a.placement)["perm"] if a.placement else {}
DUR = "DEVICE KERNEL DURATION [ns]"
df = pd.read_csv(a.csv, low_memory=False, usecols=["OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", DUR])[
    ["OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", DUR]
]
lre = re.compile(r"M3_ZONE_(START|END) layer(\d+)_sparse")
inz = False
L = None
rec = []
for row in df.itertuples(index=False):
    code = str(row[0])
    if row[1] == "signpost":
        if "M3_ZONE_START profiled_chunk" in code:
            inz = True
        elif "M3_ZONE_END profiled_chunk" in code:
            inz = False
        m = lre.search(code)
        if m:
            L = int(m.group(2)) if m.group(1) == "START" else None
        continue
    if (
        inz
        and L is not None
        and code in ("UnifiedRoutedExpertFfnDeviceOperation", "CombineDeviceOperation", "DispatchDeviceOperation")
    ):
        rec.append((L, code, int(row[3]), row[4] / 1e3))
r = pd.DataFrame(rec, columns=["L", "op", "dev", "us"])
for L in sorted(r.L.unique()):
    cnt = ep.expert_counts(cap["layers"][L][lo:hi])
    ch = ep.chip_loads(cnt, plc.get(L))  # [g, r]
    rows = {dev_ids[rr * C + g]: ch[g, rr].item() for rr in range(R) for g in range(C)}
    act = {
        dev_ids[rr * C + g]: int(
            (cnt[(plc.get(L) if plc.get(L) is not None else ep.identity())].view(C, R, 8)[g, rr] > 0).sum()
        )
        for rr in range(R)
        for g in range(C)
    }
    x = r[(r.L == L) & (r.op == "UnifiedRoutedExpertFfnDeviceOperation")].set_index("dev")["us"]
    common = [d for d in x.index if d in rows]
    xs = torch.tensor([x[d] for d in common])
    ys = torch.tensor([rows[d] for d in common])
    cc = torch.corrcoef(torch.stack([xs, ys]))[0, 1].item() if len(common) > 2 else float("nan")
    # least squares us = a + b * rows
    A = torch.stack([torch.ones_like(ys), ys], 1)
    sol = torch.linalg.lstsq(A, xs[:, None]).solution[:, 0]
    print(
        f"L{L}: rows max/mean {ch.max()/ch.mean():.3f} (max {ch.max():.0f}, mean {ch.mean():.0f}); experts us max/mean {xs.max()/xs.mean():.3f}; corr(rows, us) {cc:.3f}; fit us = {sol[0]:.0f} + {sol[1]*1000:.1f} ns/row"
    )
