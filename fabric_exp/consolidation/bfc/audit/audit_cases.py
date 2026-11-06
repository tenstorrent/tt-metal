#!/usr/bin/env python3
"""audit_cases.py ARM [ARM ...]: per P150 case (bh_*) and ERISC, how the arm changes the loop functions
(sender::run, receiver::run, single::*): 'code' = instruction count or sp-relative load/store count changes, 'sched' =
the normalized text differs at equal counts, 'none' = no loop function differs (image differences elsewhere / layout).
Counts are images per class."""
import collections, concurrent.futures, re, sys
sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
sys.argv, args = sys.argv[:1], sys.argv[1:]
import audit_cmp as ac  # noqa: E402  (its report loop runs on an empty argv)
LOOP = ac.LOOPS
def cls(pair):
    k, ea, eb = pair
    if ac.cov.image(ea) == ac.cov.image(eb):
        return k, "same", ""
    fa, fb = ac.fndiff.funcs(ea), ac.fndiff.funcs(eb)
    worst, det = "none", []
    for n in set(fa) | set(fb):
        short = re.sub(r"\(.*", "", n)
        if not LOOP.match(short) or fa.get(n) == fb.get(n):
            continue
        ta, tb = fa.get(n, ""), fb.get(n, "")
        la, lb = len(ta.splitlines()), len(tb.splitlines())
        sa, sb = ac.stack_mem(ta), ac.stack_mem(tb)
        c = "code" if (la != lb or sa != sb) else "sched"
        if c == "code" or worst == "none":
            worst = c if worst != "code" else worst
        det.append(f"{short.split('::', 1)[1]} {lb - la:+d} st {sa[0]}/{sa[1]}->{sb[0]}/{sb[1]}")
    return k, worst, "; ".join(sorted(det))
for arm in args:
    a, b = ac.elfs("a_base"), ac.elfs("a_" + arm)
    keys = sorted(k for k in set(a) & set(b) if k.startswith("bh_"))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(cls, [(k, a[k], b[k]) for k in keys]))
    per = collections.defaultdict(collections.Counter)
    dets = collections.defaultdict(collections.Counter)
    for k, c, d in res:
        ce = k.split("__")[0] + " " + ("E1" if "subordinate" in k.split("__")[-1] else "E0")
        per[ce][c] += 1
        if d:
            dets[ce][d] += 1
    print(f"== {arm}")
    for ce in sorted(per):
        c = per[ce]
        if c.get("code") or c.get("sched"):
            top = "; ".join(f"[{d}] x{n}" for d, n in dets[ce].most_common(2))
            print(f"   {ce:28s} code {c.get('code', 0):3d} sched {c.get('sched', 0):3d} none {c.get('none', 0):3d} same {c.get('same', 0):3d}  {top[:230]}")
