import re, sys
from datetime import datetime


def ts(l):
    return datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S.%f").timestamp()


rows = []
cur = None
prev_end = None
for l in open(sys.argv[1]):
    if "Running LTX AV Fast" in l:
        cur = {"t0": ts(l), "s1": 0.0}
    if cur is None:
        continue
    if "Encoding (device)" in l:
        cur["enc"] = ts(l) - cur["t0"]
        cur["te"] = ts(l)
    if "Stage 1:" in l:
        cur["s1s"] = ts(l)
    if "Stage 1 denoise" in l:
        cur["s1"] = ts(l) - cur["s1s"]
        cur["s1e"] = ts(l)
    if "Stage 2:" in l:
        cur["up"] = ts(l) - cur["s1e"]
        cur["s2s"] = ts(l)
    if "Stage 2 denoise" in l:
        cur["s2"] = ts(l) - cur["s2s"]
        cur["s2e"] = ts(l)
    if "VAE decode (forward)" in l:
        cur["vae"] = ts(l) - cur["s2e"]
        cur["ve"] = ts(l)
    if "Audio decode:" in l:
        cur["aud"] = ts(l) - cur["ve"]
        cur["ae"] = ts(l)
    if "Video export:" in l:
        cur["exp"] = ts(l) - cur["ae"]
    m = re.search(r"E2E_WALL_S gen=(\d+) seed=(\d+) wall=([\d.]+)", l)
    if m:
        cur.update(gen=int(m[1]), seed=int(m[2]), wall=float(m[3]))
        cur["other"] = cur["wall"] - sum(cur[k] for k in "enc s1 up s2 vae aud exp".split())
        rows.append(cur)
        cur = None
print("gen seed  wall   enc   s1(6) ups   s2(1) vae   audio export other")
for r in rows:
    print(
        f"{r['gen']:3d} {r['seed']:4d} {r['wall']:6.3f} "
        + " ".join(f"{r[k]:5.3f}" for k in "enc s1 up s2 vae aud exp other".split())
    )
w = [r for r in rows if r["gen"] >= 1]
import statistics as s

print(
    "mean gen1-5", " ".join(f"{k}={s.mean(r[k] for r in w):.3f}" for k in "wall enc s1 up s2 vae aud exp other".split())
)
w = [r for r in rows if r["gen"] >= 2]
print(
    "mean gen2-5", " ".join(f"{k}={s.mean(r[k] for r in w):.3f}" for k in "wall enc s1 up s2 vae aud exp other".split())
)
