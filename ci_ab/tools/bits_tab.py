#!/usr/bin/env python3
# bits_tab.py <out dir> <label>...: per case sha of the raw result tile and the element [0] value per label, then pairwise
# counts against the first label (r0 identical, full tile identical) and the mismatching cases.
import glob, hashlib, os, struct, sys
import numpy as np
out, labels = sys.argv[1], sys.argv[2:]
tab = {}
for L in labels:
    t = {}
    for f in sorted(glob.glob(f"{out}/bits_{L}/*.npz")):
        z = np.load(f); n = os.path.basename(f)[:-4]
        t[n] = (hashlib.sha256(z["res"].tobytes()).hexdigest()[:16], struct.pack("<d", float(z["r0"][0])).hex())
    tab[L] = t
    with open(f"{out}/bits_{L}.tsv", "w") as fh:
        for n, (h, r) in sorted(t.items()): fh.write(f"{n}\t{h}\t{r}\n")
a = labels[0]
for b in labels[1:]:
    common = sorted(set(tab[a]) & set(tab[b]))
    r0 = sum(tab[a][n][1] == tab[b][n][1] for n in common); full = sum(tab[a][n][0] == tab[b][n][0] for n in common)
    print(f"== bits {a} vs {b}: {len(common)} cases ({len(tab[a])}/{len(tab[b])}), element[0] identical {r0}, full tile identical {full}")
    for n in common:
        if tab[a][n][1] != tab[b][n][1]: print(f"   r0 differs {n} {tab[a][n][1]} {tab[b][n][1]}")
