#!/usr/bin/env python3
"""loopclass.py BASE NEW LIST: classifies each loop (loopbody.py's loop regions, stack offsets normalised) of every
image in LIST between two cov_m builds: identical; memory offsets only (the same instructions, an offset differs:
frame layout, also via an index register); same opcodes (registers or operand order differ); OTHER. Prints counts per
function and class, the offset shifts seen, and the first OTHER images."""
import collections, concurrent.futures as cf, re, sys
base, new, lst = sys.argv[1:4]
sys.argv = ["loopbody.py", base, new, "/dev/null", "--offsets"]
src = open(__file__.replace("loopclass.py", "loopbody.py")).read().split("cnt = collections.Counter()")[0]
ns = {"__file__": __file__.replace("loopclass.py", "loopbody.py")}
exec(src, ns)
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
IMM = re.compile(r"-?\d+(?=\()")


def one(rel):
    A, B = ns["loops"](f"{CB}/{base}/{rel}"), ns["loops"](f"{CB}/{new}/{rel}")
    out = []
    for f in sorted(set(A) | set(B)):
        a, b = A.get(f), B.get(f)
        if a == b:
            out.append((f, "identical", []))
        elif a is None or b is None or len(a) != len(b):
            out.append((f, "OTHER", []))
        elif all(IMM.sub("K", x) == IMM.sub("K", y) for x, y in zip(a, b)):
            out.append((f, "memory offsets only",
                        [int(IMM.findall(y)[0]) - int(IMM.findall(x)[0]) for x, y in zip(a, b) if x != y]))
        elif [x.split()[0] for x in a] == [x.split()[0] for x in b]:
            out.append((f, "same opcodes", []))
        else:
            out.append((f, "OTHER", []))
    return rel, out


cnt, shifts, other = collections.Counter(), collections.Counter(), []
with cf.ProcessPoolExecutor(16) as ex:
    for rel, out in ex.map(one, open(lst).read().split()):
        for f, c, s in out:
            cnt[(f, c)] += 1
            shifts.update(s)
            if c == "OTHER":
                other.append((rel, f))
for (f, c), n in sorted(cnt.items()):
    print(f"{n:5d}  {f:24s} {c}")
print("offset shifts:", dict(shifts))
for rel, f in other[:10]:
    print("  OTHER:", rel, f)
