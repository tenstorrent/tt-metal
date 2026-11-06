"""rundiff.py: per router class, the study's and the port's run() instruction streams, normalized (registers, addresses)
and diffed, with each changed instruction's addr2line inline chain; plus counts of loads / sp loads / stores."""
import collections, difflib, json, re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from annot import annotate
REG = r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b"
def norm(t): return re.sub(r"\b0x[0-9a-f]+\b|\b[0-9a-f]{4,}\b", "N", re.sub(REG, "R", t))
def mem(t):
    op = t.split()[0]
    if op in ("lw", "lh", "lhu", "lb", "lbu"): return "sp load" if "(sp)" in t else "load"
    if op in ("sw", "sh", "sb"): return "sp store" if "(sp)" in t else "store"
    return None
FN = {"active_erisc": r"sender::run[<(]", "subordinate_active_erisc": r"receiver::run[<(]"}
pairs = json.load(open(sys.argv[1]))
want = sys.argv[2] if len(sys.argv) > 2 else None
for p in pairs:
    k = p["key"]
    if not p["study"] or not p["port"] or (want and want not in str(k)):
        continue
    _, A = annotate(p["study"][0], FN[k[0]]); _, B = annotate(p["port"][0], FN[k[0]])
    ca = collections.Counter(mem(t) for _, t, _ in A if mem(t)); cb = collections.Counter(mem(t) for _, t, _ in B if mem(t))
    sm = difflib.SequenceMatcher(None, [norm(t) for _, t, _ in A], [norm(t) for _, t, _ in B], autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    nd = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    print(f"== {k}: study {len(A)} insns {dict(ca)} | port {len(B)} insns {dict(cb)} | {len(ops)} hunks, {nd} insns differ")
    for tag, i1, i2, j1, j2 in ops[: int(sys.argv[3]) if len(sys.argv) > 3 else 6]:
        print(f"   --- {tag} study[{i1}:{i2}] port[{j1}:{j2}]")
        for a, t, c in A[i1:i2][:8]: print(f"     - {t:34s} " + " < ".join(f'{f}@{l}' for f, l in c[:4])[:150])
        for a, t, c in B[j1:j2][:8]: print(f"     + {t:34s} " + " < ".join(f'{f}@{l}' for f, l in c[:4])[:150])
