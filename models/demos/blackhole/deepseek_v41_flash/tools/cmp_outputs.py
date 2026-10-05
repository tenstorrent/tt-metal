"""python cmp_outputs.py <log_a> <mode_a|-> <log_b> <mode_b|-> <scenario e.g. isl32k_b8>: compare the per-user generated 64-token OUTPUT blocks of the plain run of a scenario
(mode '-' = scenario header without a MODE suffix, else the header '... MODE <mode>:')."""
import re
import sys


def blocks(path, scen, mode):
    txt = open(path, errors="replace").read()
    pat = f"=== session scenario {scen} ===" + ("" if mode == "-" else f" MODE {mode}:")
    a = txt.find(pat)
    assert a >= 0, (path, pat)
    nxt = txt.find("=== session scenario", a + 10)
    seg = txt[a : nxt if nxt > 0 else None]
    out = {}
    for m in re.finditer(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n\n|\Z)", seg, re.S):
        out.setdefault(int(m.group(1)), m.group(2))  # first occurrence (plain run)
    first = re.search(r"First generated token: '([^']*)'", seg)
    return out, first.group(1) if first else None


la, ma, lb, mb, scen = sys.argv[1:6]
a, fa = blocks(la, scen, ma)
b, fb = blocks(lb, scen, mb)
same = [u for u in a if u in b and a[u] == b[u]]
print(f"{scen} {ma} vs {mb}: first token {fa!r} vs {fb!r}; users with identical 64-token output {len(same)}/{len(a)}")
for u in sorted(a):
    if u in b and a[u] != b[u]:
        x, y = a[u], b[u]
        k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        print(f"  user {u}: first differing char {k}/{len(x)}: A={x[max(0,k-20):k+30]!r} B={y[max(0,k-20):k+30]!r}")
        break
