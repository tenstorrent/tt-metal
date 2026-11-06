import sys, re, json, difflib, collections
sys.path.insert(0, sys.argv[1])
from annot import annotate
C = sys.argv[1]; reps = json.load(open(f"{C}/reps.json"))
REG = r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b"
def norm(t): return re.sub(r"\b[0-9a-f]{4,}\b", "ADDR", re.sub(REG, "R", t))
def loop_range(rows):
    # the pass loop: the backward branch at run@859
    for a, t, c in rows:
        if c and c[0][1].endswith(":859") and t.startswith("bnez"):
            return int(t.split(",")[-1], 16), a
    return None
def mem(t):
    op = t.split()[0]
    if op in ("lw","lh","lhu","lb","lbu"): return "sp load" if "(sp)" in t else "load"
    if op in ("sw","sh","sb"): return "sp store" if "(sp)" in t else "store"
    return None
for key in sys.argv[2:]:
    ea, eb = reps[key]["elfs"]
    _, A = annotate(ea); _, B = annotate(eb)
    la, lb = loop_range(A), loop_range(B)
    sm = difflib.SequenceMatcher(None, [norm(t) for _, t, _ in A], [norm(t) for _, t, _ in B], autojunk=False)
    tot = collections.Counter(); per = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        inl = lambda r, l: r[0] >= l[0] and not (r[2] and r[2][0][1].endswith((":867", ":862", ":855")))
        ra = [r for r in A[i1:i2] if inl(r, la)]; rb = [r for r in B[j1:j2] if inl(r, lb)]
        if not ra and not rb: continue
        fa = collections.Counter(c[0][0] if c else "?" for _, _, c in ra); fb = collections.Counter(c[0][0] if c else "?" for _, _, c in rb)
        ma = collections.Counter(mem(t) for _, t, _ in ra if mem(t)); mb = collections.Counter(mem(t) for _, t, _ in rb if mem(t))
        outer = (rb or ra)[0][2]
        ctx = " < ".join(f"{f}@{l}" for f, l in outer[:3])
        per.append(f"   {len(rb)-len(ra):+d} insns  idoff[{', '.join(t for _, t, _ in ra)}]  id[{', '.join(t for _, t, _ in rb)}]  @ {ctx[:120]}")
        tot["insns"] += len(rb) - len(ra)
        for k in set(ma) | set(mb): tot[k] += mb[k] - ma[k]
    print(f"== {key}: run() {len(A)} -> {len(B)} insns; pass loop {la[1]-la[0]+4 if la else '?'}B -> {lb[1]-lb[0]+4 if lb else '?'}B; changes in the pass loop and its out-of-line paths: {dict(tot)}")
    for p in per: print(p)
