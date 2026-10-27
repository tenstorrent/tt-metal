"""Device KDA state vs the exact fp64 recurrence on the captured inputs at every 5120 boundary, layers 0-2 (their inputs
are the same in every run), for each capture dir given."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from emu_kda import inputs, rel, run

SP = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1].split(",")
for layer in (0, 1, 2):
    q, k, v, sig, beta = inputs(f"{SP}/kc_fork", layer)
    snaps = tuple(range(5120, 56321, 5120))
    _, ex = run(q, k, v, sig, beta, [], torch.float64, snaps)
    for t in tags:
        row = []
        for p in snaps:
            d = torch.load(f"{SP}/{t}/state_{p}.pt")[layer]["kda_recurrent"].double().reshape(ex[p].shape)
            hr = max(rel(d[h], ex[p][h]) for h in range(d.shape[0]))
            row.append(f"{p // 1024}k {rel(d, ex[p]):.3f}/{hr:.2f}")
        print(f"[curve] L{layer} {t:10s} " + " ".join(row), flush=True)
