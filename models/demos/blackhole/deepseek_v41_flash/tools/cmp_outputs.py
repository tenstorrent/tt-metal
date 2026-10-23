"""python cmp_outputs.py <log A> <log B> <scenario id> -- compares the plain "==USER n - OUTPUT" texts of one session scenario of two demo logs:
number of identical users and, per differing user, the length of the common prefix (chars) / total chars."""

import re
import sys


def grab(log, scen):
    txt = open(log).read()
    txt = txt[txt.index(f"=== session scenario {scen} ===") :]
    nxt = txt.find("=== session scenario ", 10)
    txt = txt if nxt < 0 else txt[:nxt]
    out = {}
    for m in re.finditer(r"==USER (\d+) - OUTPUT\n(.*?)(?=\n==USER|\n==REPEAT|\n\d{4}-\d\d-\d\d |\Z)", txt, re.S):
        out[int(m.group(1))] = m.group(2)
    ttft = re.search(r"TTFT \(whole batch of (\d+) users, ISL max (\d+)\): (\d+) ms -> prefill (\d+) tok/s", txt)
    return out, ttft


a, ta = grab(sys.argv[1], sys.argv[3])
b, tb = grab(sys.argv[2], sys.argv[3])
users = sorted(set(a) & set(b))
same = 0
for u in users:
    if a[u] == b[u]:
        same += 1
        continue
    n = 0
    while n < min(len(a[u]), len(b[u])) and a[u][n] == b[u][n]:
        n += 1
    print(f"user {u}: common prefix {n} chars of {len(a[u])}/{len(b[u])}")
print(f"{sys.argv[3]}: {same}/{len(users)} users identical")
print("A:", ta.group(0) if ta else None)
print("B:", tb.group(0) if tb else None)
