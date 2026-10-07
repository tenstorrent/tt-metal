"""Name job A's dumped decode-input latents by seed (host only).

Each dump line is attributed to the last ``noise_seed=`` line before it (noise_seed = 10000 + seed).
The warm replay (gen >= 1) is the reference; the gen#0 capture of seed 0 must match it bit for bit.
"""

import re
import shutil
import sys
from pathlib import Path

import torch

log, raw, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
seed, gen, found = None, 0, []
for line in log.read_text(errors="replace").splitlines():
    if m := re.search(r"noise_seed=(\d+)", line):
        seed = int(m.group(1)) - 10000
    if m := re.search(r"steady-state pass \(gen #(\d+), seed (\d+)", line):
        gen = int(m.group(1))
    if m := re.search(r"dumped decode-input latent to (\S+)", line):
        found.append((gen, seed, Path(m.group(1))))
print(f"{len(found)} dumps: {[(g, s, p.name) for g, s, p in found]}")
by_seed = {}
for gen, seed, path in found:
    if gen >= 1:
        by_seed[seed] = path
missing = [s for s in range(5) if s not in by_seed]
if missing:
    sys.exit(f"no warm-replay latent for seeds {missing}")
cap = [p for g, s, p in found if g == 0 and s == 0]
if cap:
    same = torch.equal(torch.load(cap[-1]), torch.load(by_seed[0]))
    print(f"seed 0 capture vs warm replay latent identical: {same}")
for s, p in sorted(by_seed.items()):
    lat = torch.load(p)
    shutil.copyfile(p, dst / f"seed{s}.pt")
    print(
        f"seed{s}.pt <- {p.name} shape={tuple(lat.shape)} dtype={lat.dtype} mean={lat.mean():.5f} std={lat.std():.5f}"
    )
