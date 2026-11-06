import sys, re, difflib
def load(p):
    rows = []
    for l in open(p):
        a, rest = l[:6], l[8:].rstrip("\n")
        ins, chain = rest[:40].strip(), rest[40:].strip()
        op = ins.split()[0] if ins else ""
        norm = re.sub(r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b", "R", ins)
        norm = re.sub(r"\b[0-9a-f]{4,}\b", "ADDR", norm)
        rows.append((a.strip(), ins, chain, norm))
    return rows
A, B = load(sys.argv[1]), load(sys.argv[2])
sm = difflib.SequenceMatcher(None, [r[3] for r in A], [r[3] for r in B], autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal": continue
    print(f"--- {tag} base[{i1}:{i2}] new[{j1}:{j2}]")
    for r in A[i1:i2]: print(f"  - {r[0]} {r[1]:34s} {r[2][:150]}")
    for r in B[j1:j2]: print(f"  + {r[0]} {r[1]:34s} {r[2][:150]}")
