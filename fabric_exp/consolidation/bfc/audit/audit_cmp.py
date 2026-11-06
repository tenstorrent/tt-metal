#!/usr/bin/env python3
"""audit_cmp.py ARM [ARM ...]: each placement-audit arm's router images (offline build a_<arm>) against the base build
(a_base), both from the credit capture's commands. Per arm: identical / differing images (loadable bytes); for the
differing ones, the functions whose code differs (branch/call targets normalized), each function's instruction count
base -> arm, and the loop functions' (sender::run, receiver::run, single::*_pass) stack loads and stores (sp-relative
lw/sw) base -> arm. With 'jit', the base build against the capture's own JIT images (a check of the PCH-less build)."""
import collections, concurrent.futures, glob, os, re, sys

sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
sys.path.insert(0, "/home/aagarwal/tt-metal-mainline/.scratch")
os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-mainline-cov-bfc-rjcred")
os.environ.setdefault("TT_COV_TREE", "/home/aagarwal/tt-metal-mainline-glx")
import cov  # noqa: E402
import fndiff  # noqa: E402

LOOPS = re.compile(r"^fabric_router::(sender::run|receiver::run|single::sender_pass|single::receiver_pass|single::run)\b")


def elfs(name):
    return cov.elfs_of(name)


def stack_mem(text):
    loads = sum(1 for l in text.splitlines() if re.match(r"l[bhw]u?\s", l) and "(sp)" in l)
    stores = sum(1 for l in text.splitlines() if re.match(r"s[bhw]\s", l) and "(sp)" in l)
    return loads, stores


def one(pair):
    k, ea, eb = pair
    if cov.image(ea) == cov.image(eb):
        return k, None
    fa, fb = fndiff.funcs(ea), fndiff.funcs(eb)
    ch = {}
    for n in set(fa) | set(fb):
        if fa.get(n) != fb.get(n):
            short = re.sub(r"\(.*", "", n)
            ta, tb = fa.get(n, ""), fb.get(n, "")
            ch[short] = (len(ta.splitlines()), len(tb.splitlines()),
                         stack_mem(ta) if LOOPS.match(short) else None, stack_mem(tb) if LOOPS.match(short) else None)
    return k, ch


def report(arm):
    if arm == "jit":
        a = {}
        for case in os.listdir(os.path.join(cov.COV, "cases")):
            a.update({k: v for k, v in cov.elfs_of("jit:" + case).items() if "__fabric_erisc_router__" in k})
        b = elfs("a_base")
        label = "a_base vs the capture's JIT images"
    else:
        a, b = elfs("a_base"), elfs("a_" + arm)
        label = f"{arm} vs base"
    shared = sorted(set(a) & set(b))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(one, [(k, a[k], b[k]) for k in shared]))
    diff = [(k, ch) for k, ch in res if ch is not None]
    print(f"== {label}: shared {len(shared)}, identical {len(shared) - len(diff)}, differ {len(diff)}, "
          f"only base {len(set(a) - set(b))}, only arm {len(set(b) - set(a))}")
    if not diff:
        return
    per_case = collections.Counter(k.split("__")[0] + " " + k.split("__")[-1] for k, _ in diff)
    print("   differing by case/ERISC:", ", ".join(f"{c} {n}" for c, n in sorted(per_case.items())))
    fns = collections.defaultdict(collections.Counter)
    for k, ch in diff:
        for f, (x, y, sa, sb) in ch.items():
            fkey = re.sub(r"<\d+u>", "<N>", f)
            extra = f" stack ld/st {sa[0]}/{sa[1]} -> {sb[0]}/{sb[1]}" if sa else ""
            fns[fkey][f"{x}->{y} ({y - x:+d}){extra}"] += 1
    for f, c in sorted(fns.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"   {f}: " + "; ".join(f"{v} x{n}" for v, n in c.most_common(6)) + ("; ..." if len(c) > 6 else ""))


for arm in sys.argv[1:]:
    report(arm)
