# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static per-layer expert placement from measured routing counts (test_expert_counts.py's counts .pt): per MoE layer,
the experts placed on the chips by greedy longest-first (the hottest expert onto the currently least-loaded chip
with a free slot; experts_per_chip slots each) on the counts of the calibration chunks. Writes the JSON
MiMoRuntimeOptions.expert_placement (MIMO_EXPERT_PLACEMENT) reads, and prints the calibration / held-out
chip-load max/mean before and after.

    python models/demos/mimo_v2_d_p/tests/perf/expert_placement.py <counts.pt> <out.json> [calib_chunks, e.g. 0-6]
"""

import json
import sys

import torch


def lpt(weights, n_dev, epc):
    load, slots = [0.0] * n_dev, [[] for _ in range(n_dev)]
    for e in weights.argsort(descending=True).tolist():
        d = min((d for d in range(n_dev) if len(slots[d]) < epc), key=lambda d: load[d])
        slots[d].append(e)
        load[d] += float(weights[e])
    return [sorted(s) for s in slots]


def imbalance(G, gids):
    """G [chunks, E] counts -> per chunk chip-load max/mean of placement gids."""
    loads = torch.stack([G[:, g].sum(1) for g in gids], 1)
    return loads.max(1).values / loads.mean(1)


def main():
    d = torch.load(sys.argv[1])
    C, layers = d["counts"].double(), d["moe_layers"]
    LG = d.get("layer_gids", d["gids"].expand(len(layers), *d["gids"].shape)).tolist()
    n_ch, n_l, n_dev, epc = C.shape
    E = n_dev * epc
    a, b = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else f"0-{n_ch // 2 - 1}").split("-"))
    calib = list(range(a, b + 1))
    held = [c for c in range(n_ch) if c not in calib] or calib
    out, rows = {}, []
    for m, li in enumerate(layers):
        G = torch.zeros(n_ch, E, dtype=torch.float64)
        gids0 = LG[m]
        for dv in range(n_dev):
            G[:, gids0[dv]] = C[:, m, dv]
        new = lpt(G[calib].mean(0), n_dev, epc)
        out[str(li)] = new
        rows.append((imbalance(G[held], gids0).mean().item(), imbalance(G[held], new).mean().item()))
    json.dump({"mesh": list(d["mesh"]), "calib_chunks": calib, "layers": out}, open(sys.argv[2], "w"))
    r = torch.tensor(rows)
    print(
        f"{len(out)} layers, calibrated on chunks {calib}; held-out chunks {held} chip-load max/mean: "
        f"{r[:, 0].mean():.2f}x -> {r[:, 1].mean():.2f}x (worst layer {r[:, 0].max():.2f}x -> {r[:, 1].max():.2f}x)"
    )


if __name__ == "__main__":
    main()
