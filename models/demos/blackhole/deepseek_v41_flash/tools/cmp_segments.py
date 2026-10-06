#!/usr/bin/env python3
"""python cmp_segments.py <session log> -- splits a demo session log at '=== session scenario' markers and prints, per scenario segment: id, flags, TTFT, prefill tok/s,
decode ms/token, first generated token, and for every segment after the first of the same id the number of users whose generated text is identical to the first segment (and the common-prefix chars of the differing ones)."""
import re
import sys

txt = open(sys.argv[1]).read()
parts = re.split(r"(?=\S* ?- === session scenario )", txt)
segs = [p for p in parts if "=== session scenario" in p[:200]]
first = {}
seen = {}
for s in segs:
    m = re.search(r"=== session scenario (\S+) \((.*?)\) ===(.*?MODE (\S+))?", s)
    sid, flags, mode = m.group(1), m.group(2), m.group(4) or "-"
    t = re.search(r"TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms -> prefill (\d+) tok/s", s)
    d = re.search(r"Decode: ([\d.]+) ms/token", s)
    ft = re.findall(r"First generated token: (.*)", s)
    out = {
        int(u): o
        for u, o in re.findall(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{4}-\d\d-\d\d |\Z)", s, re.S)
    }
    line = (
        f"{sid:12s} {flags} mode={mode}: "
        + (f"TTFT {int(t.group(3))/1000:.2f} s, {t.group(4)} tok/s" if t else "no TTFT")
        + (f", decode {d.group(1)} ms/tok" if d else "")
    )
    seen.setdefault(sid, []).append(out)
    if (
        len(seen[sid]) >= 3
    ):  # run-to-run determinism: same output as the scenario two entries earlier (same configuration in an A/B/A/B session)
        o2 = seen[sid][-3]
        line += f" | repeat-vs-2-earlier: {sum(1 for u in out if out[u] == o2.get(u))}/{len(out)} identical"
    if sid not in first:
        first[sid] = (out, ft)
    else:
        o0, f0 = first[sid]
        same = sum(1 for u in out if u in o0 and out[u] == o0[u])
        line += f" | vs first {sid}: {same}/{len(out)} users identical, first tokens {'same' if ft == f0 else 'DIFFER'}"
        for u in out:
            if u in o0 and out[u] != o0[u]:
                n = 0
                while n < min(len(out[u]), len(o0[u])) and out[u][n] == o0[u][n]:
                    n += 1
                line += f" [u{u}: common {n}/{len(o0[u])} chars]"
    print(line)
