#!/usr/bin/env python3
"""usage: rc_compare.py <session.log> <fresh_dir_or_prefix> [tag]   (fresh logs: <prefix><scenario>.log, one fresh process per scenario)
Splits a multi-scenario session log (DSV41_SESSION) at '=== session scenario <id>' and compares, per scenario, the first generated tokens (FIRSTTOK_IDS) and the
decoded text of every user (the '==USER i - OUTPUT' blocks) with the fresh-process log of the same scenario. Prints one line per scenario segment."""
import re
import sys

sess, prefix = sys.argv[1], sys.argv[2]


def parse(text):
    first = re.findall(r"FIRSTTOK_IDS (\[.*?\])", text)
    outs = re.findall(r"==USER (\d+) - OUTPUT\n(.*?)\n\n", text, flags=re.S)
    return first[-1] if first else None, {int(u): o.strip() for u, o in outs}


txt = open(sess, errors="replace").read()
parts = re.split(r"=== session scenario (\S+) \(", txt)
ok_all = True
for i in range(1, len(parts), 2):
    sc, body = parts[i], parts[i + 1]
    try:
        ftxt = open(f"{prefix}{sc}.log", errors="replace").read()
    except FileNotFoundError:
        print(f"{sc}: no fresh log {prefix}{sc}.log")
        continue
    f1, o1 = parse(body)
    f2, o2 = parse(ftxt)
    same_first = f1 == f2 and f1 is not None
    n = len(o1)
    eq = sum(1 for u in o1 if o1[u] == o2.get(u))
    ok = same_first and n > 0 and eq == n == len(o2)
    ok_all &= ok
    print(
        f"{sc}: first tokens {'IDENTICAL' if same_first else 'DIFFER'} ({f1} vs {f2}); text identical for {eq}/{n} users (fresh has {len(o2)}) -> {'OK' if ok else 'MISMATCH'}"
    )
sys.exit(0 if ok_all else 1)
