import re
import sys

pat = r"lg_md5=\{(\d+): '(\w+)'\} pfx_start=(\d+) hits=\{\d+: (\d+)\}.*?mode=(\w+).*?lens=\[(\d+)\].*?prefill=([\d.]+)s"
calls = []
for l in open(sys.argv[1], errors="replace"):
    if "HY4 prefill call" in l:
        m = re.search(pat, l)
        if m:
            calls.append(
                dict(row=int(m[1]), md5=m[2], start=int(m[3]), hit=int(m[4]), mode=m[5], n=int(m[6]), pf=float(m[7]))
            )
calls = [c for c in calls if c["n"] != 1024 or True][int(sys.argv[2]) :]
i = 0
while i + 3 < len(calls):
    a, b, w, h = calls[i : i + 4]
    ok = a["n"] == b["n"] == h["n"] and a["md5"] == b["md5"] == h["md5"] and h["start"] > 0
    print(
        f"E={a['n']} cold rows {a['row']},{b['row']} md5 {a['md5']} {b['md5']} | warm B n={w['n']} row {w['row']} start {w['start']} | hit row {h['row']} start {h['start']} md5 {h['md5']} prefill {h['pf']}s vs cold {a['pf']}s -> {'MD5_EQUAL' if ok else 'DIFF'}"
    )
    i += 4
