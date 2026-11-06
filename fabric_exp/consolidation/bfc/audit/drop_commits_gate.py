#!/usr/bin/env python3
"""drop_commits_gate.py: gate of the drop commits on the cleanup branch, from their offline builds (a_c<sha>), all from
the credit capture. Each step (commit vs the one before; the first vs a_dlvcmt = c7a4d0fb29f, image-identical to
7cd3c13a0db) must change exactly the functions its single-site arm changed from base (a_<arm> vs a_base), per image;
the comment-only commit must be image-identical to the one before; the last commit must equal a_drop5 exactly."""
import collections, concurrent.futures, re, sys

sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
sys.argv, _a = sys.argv[:1], sys.argv[1:]
import audit_cmp as ac  # noqa: E402

STEPS = [("993c99a564b", "rg_s537"), ("aa11a8786df", "rg_s583"), ("82ea91f44a4", "rg_r755"),
         ("5fb378a7880", "kim_s816"), ("bc0108869e2", "kim_r713"), ("22cdb1e55c3", None)]
short = lambda n: re.sub(r"\(.*", "", n)
E = {n: ac.elfs(n) for n in ["a_base", "a_dlvcmt", "a_drop5"] + [f"a_c{c}" for c, _ in STEPS]
     + [f"a_{a}" for _, a in STEPS if a]}
keys = sorted(set.intersection(*(set(v) for v in E.values())))
print(f"images {len(keys)} in every build")


def changed(a, b):
    if ac.cov.image(a) == ac.cov.image(b):
        return frozenset()
    fa = {short(n): t for n, t in ac.fndiff.funcs(a).items()}
    fb = {short(n): t for n, t in ac.fndiff.funcs(b).items()}
    return frozenset(n for n in set(fa) | set(fb) if fa.get(n) != fb.get(n))


prev = "a_dlvcmt"
for c, arm in STEPS:
    cur = f"a_c{c}"
    def one(k, prev=prev, cur=cur, arm=arm):
        got = changed(E[prev][k], E[cur][k])
        want = changed(E["a_base"][k], E[f"a_{arm}"][k]) if arm else frozenset()
        return k, got, want
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(one, keys))
    differ = sum(1 for _, g, _ in res if g)
    bad = [(k, g, w) for k, g, w in res if g != w]
    print(f"== {c} ({arm or 'comment only'}): {differ} images change; function sets as the single arm's in "
          f"{len(res) - len(bad)} / {len(res)}")
    summ = collections.Counter()
    for k, g, w in bad:
        ce = k.split("__")[0] + " " + ("E1" if "subordinate" in k.split("__")[-1] else "E0")
        summ[(ce, f"extra {sorted(g - w)} missing {sorted(w - g)}")] += 1
    for (ce, d), n in sorted(summ.items())[:20]:
        print(f"   {ce:28s} {d}: {n}")
    prev = cur
same = sum(ac.cov.image(E[prev][k]) == ac.cov.image(E["a_drop5"][k]) for k in keys)
print(f"== last commit vs a_drop5: identical {same} / {len(keys)}")
