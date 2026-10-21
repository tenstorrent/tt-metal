"""Collect the traced-prefill context table (first chunk = 1k row, 100k, 256k) from every e2e log in this folder.

    python3 report.py            # markdown tables to stdout
Tags are <variant>_c<chunk>_r<rep>; repeats are averaged and their spread reported.
"""
import collections
import glob
import os
import re
import statistics as st

W = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s*(1k|100k|256k)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")
data = collections.defaultdict(lambda: collections.defaultdict(list))  # (variant, chunk) -> ctx -> [(wall, dev)]
chunks_of = {}
for log in sorted(glob.glob(os.path.join(W, "*_c*_r*.log"))):
    tag = os.path.basename(log)[:-4]
    if tag.endswith(".build"):
        continue
    m = re.match(r"(\w+?)_c(\d+)_r(\d+)$", tag)
    if not m:
        continue
    variant, chunk = m.group(1), int(m.group(2))
    lines = open(log, errors="replace").read().splitlines()
    try:
        i = max(k for k, l in enumerate(lines) if "context wall and device times" in l)
    except ValueError:
        continue
    for l in lines[i + 1 : i + 8]:
        r = ROW.match(l)
        if r:
            data[(variant, chunk)][r.group(1)].append((float(r.group(3)), float(r.group(4))))
            chunks_of[(chunk, r.group(1))] = int(r.group(2))

NAMES = {
    "base": "main (909748641c1)",
    "vfk": "v-from-k",
    "ovl": "v-from-k-overlap",
    "ks8": "ksplit8-q128",
    "combo": "ksplit8-q128 + v-from-k",
}
for kind, idx in (("Device", 1), ("Wall", 0)):
    print(f"\n### {kind} time (ms), mean of repeats (spread)\n")
    print("| Chunk | Variant | First chunk | 100k | 256k | 256k vs main |")
    print("|---:|---|---:|---:|---:|---:|")
    for chunk in (2048, 4096, 8192):
        base256 = None
        for v in ("base", "vfk", "ovl", "ks8", "combo"):
            d = data.get((v, chunk))
            if not d:
                print(f"| {chunk} | {NAMES[v]} | — | — | — | — |")
                continue
            cells = []
            for ctx in ("1k", "100k", "256k"):
                vals = [x[idx] for x in d.get(ctx, [])]
                if not vals:
                    cells.append("—")
                    continue
                mean = st.mean(vals)
                spread = max(vals) - min(vals)
                cells.append(f"{mean:.1f} (±{spread / 2:.1f})" if len(vals) > 1 else f"{mean:.1f}")
                if ctx == "256k":
                    m256 = mean
            if v == "base":
                base256 = m256
            delta = "—" if v == "base" or base256 is None else f"{(m256 / base256 - 1) * 100:+.1f}%"
            print(f"| {chunk} | {NAMES[v]} | {cells[0]} | {cells[1]} | {cells[2]} | {delta} |")
print("\nChunks per context (rounded up):", {f"{c}/{k}": n for (c, k), n in sorted(chunks_of.items())})

print("\n### KV-cache PCC (test_prefill_migration[mock-256k])\n")
print("| Chunk | Variant | Overall PCC (>= 0.978) | Min per-head PCC (>= 0.928) | Result |")
print("|---:|---|---:|---:|---|")
for log in sorted(glob.glob(os.path.join(W, "*_kvpcc_c*.log"))):
    tag = os.path.basename(log)[:-4]
    m = re.match(r"(\w+?)_kvpcc_c(\d+)$", tag)
    if not m:
        continue
    t = open(log, errors="replace").read()
    o = re.findall(r"Overall PCC\s*│[^│]*│\s*([\d.]+)", t)
    mn = re.findall(r"Minimum per-head PCC\s*│[^│]*│\s*([\d.]+)", t)
    res = "PASS" if re.search(r"1 passed", t) else ("FAIL" if re.search(r"failed", t) else "?")
    print(
        f"| {m.group(2)} | {NAMES.get(m.group(1), m.group(1))} | {o[-1] if o else '—'} | {mn[-1] if mn else '—'} | {res} |"
    )
