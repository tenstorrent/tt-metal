# Usage: gen_table.py <matrix_vs_main logs...>: markdown table of main vs expanded V vs expanded KV from the RESULT
# lines of matrix_vs_main.py (main's 2k cells use baselineTuned when present).
import re
import sys

cells = {}
for path in sys.argv[1:]:
    for line in open(path, errors="ignore"):
        m = re.search(r"RESULT (\w+) (\w+) chunk=(\d+) (\w+) kv=(\d+)(?: meta=\w+)?: ([\d.]+) ms", line)
        if m:
            mode, model, chunk, fid, kv, ms = m.groups()
            cells[(mode, model, int(chunk), fid, int(kv))] = float(ms)


def get(mode, *key):
    return cells.get((mode, *key))


kvs = sorted({k[4] for k in cells})
print("| Model | Chunk | Fidelity | " + " | ".join(f"KV {kv:,}: main / exp V / exp KV / vs main" for kv in kvs) + " |")
print("|---|---|---|" + "---|" * len(kvs))
for model in ("kimi_k3", "kimi_k2_7"):
    for chunk in (5120, 2048):
        for fid in ("HiFi2", "LoFi"):
            row = []
            for kv in kvs:
                key = (model, chunk, fid, kv)
                base = (get("baselineTuned", *key) if chunk == 2048 else None) or get("baseline", *key)
                ev, ekv = get("expandedV", *key), get("expandedKV", *key)
                fmt = lambda v: f"{v:.2f}" if v else "-"
                ratio = f"{base / ekv:.2f}x" if base and ekv else "-"
                row.append(f"{fmt(base)} / {fmt(ev)} / {fmt(ekv)} / {ratio}")
            if any(r != "- / - / - / -" for r in row):
                print(f"| {model} | {chunk // 1024}k | {fid} | " + " | ".join(row) + " |")
