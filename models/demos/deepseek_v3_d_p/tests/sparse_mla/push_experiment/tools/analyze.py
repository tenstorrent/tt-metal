"""analyze.py <tagA> <tagB> [chunk=5120]: per-context host chunk time A vs B, and per-op RT device medians per context band."""
import os
import glob
import json
import statistics as st
import sys
from collections import defaultdict

S = os.environ.get("PUSH_EXP_DIR", "/tmp/push_exp") + "/runs"
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
BANDS = [0, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 2_000_000]


def curve(tag):
    import re
    rx = re.compile(r"CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=([\d.]+) warmup=([01])")
    spans = {}
    for line in open(f"{S}/{tag}/launch.log", errors="replace"):
        m = rx.search(line)
        if m and m.group(6) == "0":
            spans[int(m.group(2))] = (float(m.group(4)), float(m.group(5)))
    if not spans:
        return None
    ks = sorted(spans)
    # per chunk: retire(i) - retire(i-1) (steady-state chunk period, incl. layer 0), first chunk: retire - first
    out = [ (spans[ks[0]][1] - spans[ks[0]][0]) * 1e3 ]
    out += [ (spans[ks[i]][1] - spans[ks[i-1]][1]) * 1e3 for i in range(1, len(ks)) ]
    return out


def cat(path):
    for key, name in (
        ("sparse_kv_push/plan", "plan"),
        ("sparse_kv_push/push", "push"),
        ("high_bw_all_gather", "hbag"),
        ("topk_large_indices", "topk"),
        ("indexer_score", "indexer"),
        ("sparse_sdpa", "sdpa"),
    ):
        if key in path:
            return name
    return None


def rt(tag, n_chunks):
    """-> {op: [per-chunk list of per-occurrence max-across-chips durations (us)]}"""
    f = glob.glob(f"{S}/{tag}/rt/rank*.json")
    if not f:
        return None
    d = json.load(open(f[0]))
    freq = d["meta"]["freq"]  # GHz: cycles per ns
    names = {int(k): v for k, v in d["names"].items()}
    per = defaultdict(lambda: defaultdict(list))  # op -> chip -> [(start, dur)]
    for chip, rid, s, e in d["recs"]:
        c = cat(names.get(rid, ""))
        if c:
            per[c][chip].append((s, (e - s) / freq / 1e3))
    out = {}
    for op, chips in per.items():
        seqs = [sorted(v) for v in chips.values()]
        n = min(len(s) for s in seqs)
        occ = [max(s[len(s) - n + i][1] for s in seqs) for i in range(n)]  # align from the end
        k = n // n_chunks if n_chunks else 0
        if k == 0:
            out[op] = (k, [])
            continue
        occ = occ[-k * n_chunks :]
        out[op] = (k, [occ[i * k : (i + 1) * k] for i in range(n_chunks)])
    return out, d["meta"]


def band_rows(values_by_chunk):
    rows = []
    for lo, hi in zip(BANDS, BANDS[1:]):
        idx = [i for i in range(len(values_by_chunk)) if lo <= i * CHUNK < hi]
        if idx:
            rows.append((lo, hi, idx))
    return rows


a, b = sys.argv[1], sys.argv[2]
ca, cb = curve(a), curve(b)
print(f"host chunk time (first->last layer ack), median per context band: {a} vs {b}")
if ca and cb:
    n = min(len(ca), len(cb))
    for lo, hi, idx in band_rows(ca[:n]):
        ma, mb = st.median(ca[i] for i in idx), st.median(cb[i] for i in idx)
        print(f"  [{lo/1e3:5.0f}k,{min(hi, n*CHUNK)/1e3:5.0f}k) n={len(idx):3d}  {ma:8.3f}  {mb:8.3f}  delta {mb-ma:+7.3f} ms ({(mb/ma-1)*100:+5.1f}%)")
    # finest crossover estimate: first chunk after which a 9-chunk rolling median of (b - a) stays < 0
    diff = [cb[i] - ca[i] for i in range(n)]
    roll = [st.median(diff[max(0, i - 4) : i + 5]) for i in range(n)]
    cross = next((i for i in range(n) if all(r < 0 for r in roll[i:])), None)
    print(f"  crossover (rolling-9 median of B-A stays < 0 from): {cross * CHUNK if cross is not None else 'never'} tokens")
    print(f"  total request: {sum(ca[:n]):.0f} vs {sum(cb[:n]):.0f} ms")
else:
    print("  missing curve:", a if not ca else b)
for tag, c in ((a, ca), (b, cb)):
    r = rt(tag, len(c) if c else 0)
    if not r:
        continue
    ops, meta = r
    print(f"\nRT device per op ({tag}; max across chips, median per band, us; dropped={meta['dropped']})")
    for op, (k, chunks) in sorted(ops.items()):
        if not chunks:
            continue
        for j in range(k):
            line = f"  {op}[{j}/{k}]"
            for lo, hi, idx in band_rows(chunks):
                line += f" {lo//1000}k:{st.median(chunks[i][j] for i in idx):7.0f}"
            print(line)
