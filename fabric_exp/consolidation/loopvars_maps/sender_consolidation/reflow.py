#!/usr/bin/env python3
"""reflow.py FILE LINE...: rewraps, at 120 columns, the comment paragraph holding each LINE (1-based): the run of
`// text` lines around it with the same indent, stopping at blank `//` lines, list items (`//   - ...`) and code."""
import re, sys
p = sys.argv[1]
lines = open(p).read().split("\n")
PLAIN = re.compile(r"^(\s*// )(?! *- )(\S.*)$")


def para(i):
    m = PLAIN.match(lines[i])
    assert m, (i + 1, lines[i])
    pre = m.group(1)
    a = i
    while a > 0 and PLAIN.match(lines[a - 1]) and PLAIN.match(lines[a - 1]).group(1) == pre:
        a -= 1
    b = i
    while b + 1 < len(lines) and PLAIN.match(lines[b + 1]) and PLAIN.match(lines[b + 1]).group(1) == pre:
        b += 1
    return a, b, pre


done = set()
for n in sorted({int(x) for x in sys.argv[2:]}, reverse=True):
    a, b, pre = para(n - 1)
    if (a, b) in done:
        continue
    done.add((a, b))
    words = " ".join(PLAIN.match(l).group(2) for l in lines[a:b + 1]).split()
    out, cur = [], pre
    for w in words:
        if len(cur) + len(w) + (0 if cur == pre else 1) > 120:
            out.append(cur)
            cur = pre + w
        else:
            cur = cur + ("" if cur == pre else " ") + w
    out.append(cur)
    lines[a:b + 1] = out
open(p, "w").write("\n".join(lines))
