#!/usr/bin/env python3
"""opcmp.py BUILD_A BUILD_B LIST FUNC...: for each image in LIST and each function whose name starts with one of FUNC,
classifies the difference between two cov_m builds: identical; "same opcodes" (equal length and the same opcode at every
position: only registers, operand order or immediates differ); "frame" (as "same opcodes" after dropping stack-pointer
offsets and init stores of constants into the frame: the same instructions plus or minus frame setup); OTHER."""
import collections, re, sys
src = open(__file__.replace("opcmp.py", "fnsplit.py")).read()
ns = {}
exec(src.split("def one(rel)")[0], ns)
funcs = ns["funcs"]
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
A, B, lst = sys.argv[1:4]
prefixes = sys.argv[4:]
cnt = collections.Counter()
ex = {}
for rel in open(lst).read().split():
    fa, fb = funcs(f"{CB}/{A}/{rel}"), funcs(f"{CB}/{B}/{rel}")
    for k in sorted(set(fa) & set(fb)):
        if not any(k.startswith(p) for p in prefixes):
            continue
        a, b = fa[k], fb[k]
        if a == b:
            c = "identical"
        elif len(a) == len(b) and [x.split()[0] for x in a] == [x.split()[0] for x in b]:
            c = "same opcodes"
        else:
            strip = lambda L: [x for x in L if not re.match(r"(sw|sh|sb) zero,\d+\(sp\)", x)]
            sa, sb = strip(a), strip(b)
            c = ("frame" if len(sa) == len(sb) and [x.split()[0] for x in sa] == [x.split()[0] for x in sb]
                 else "OTHER")
        name = re.sub(r"\(.*", "", k)
        cnt[(name, c)] += 1
        ex.setdefault((name, c), rel)
for (name, c), n in sorted(cnt.items()):
    print(f"{n:5d}  {name:45s} {c}  (e.g. {ex[(name, c)].split('/')[0]})")
