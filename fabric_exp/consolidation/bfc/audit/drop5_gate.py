#!/usr/bin/env python3
"""drop5_gate.py: the joint-drop arm (drop5 = rg_s537 + rg_s583 + rg_r755 + kim_s816 + kim_r713) against the five
single-arm builds, all offline from the credit capture (a_base = 7cd3c13a0db). Per router image:
  - changed functions (normalized text differs from base): drop5's set vs the union of the singles' sets;
  - function presence (inlining): functions added / removed in drop5 vs those any single adds / removes;
  - for the pass-loop functions (sender::run, receiver::run, single::*), instruction count and sp-relative load / store
    counts: base, each single's delta, the additive prediction base + sum of deltas, and drop5's actual.
Summary per (config, ERISC, function): images where drop5 equals the prediction, and the mismatches."""
import collections, concurrent.futures, re, sys

sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import audit_cmp as ac  # noqa: E402

SINGLES = ["rg_s537", "rg_s583", "rg_r755", "kim_s816", "kim_r713"]
LOOP = ac.LOOPS
short = lambda n: re.sub(r"\(.*", "", n)


def counts(text):
    if text is None:
        return None
    ld, st = ac.stack_mem(text)
    return (len(text.splitlines()), ld, st)


def one(k):
    b = E["a_base"][k]
    d = E["a_drop5"][k]
    img_b = ac.cov.image(b)
    singles_differ = {s: ac.cov.image(E["a_" + s][k]) != img_b for s in SINGLES}
    drop_differs = ac.cov.image(d) != img_b
    if not drop_differs and not any(singles_differ.values()):
        return k, None
    fb = {short(n): t for n, t in ac.fndiff.funcs(b).items()}
    fd = {short(n): t for n, t in ac.fndiff.funcs(d).items()}
    fs = {s: ({short(n): t for n, t in ac.fndiff.funcs(E["a_" + s][k]).items()} if singles_differ[s] else fb)
          for s in SINGLES}
    chg = lambda f: {n for n in set(fb) | set(f) if fb.get(n) != f.get(n)}
    cd = chg(fd)
    cs = {s: chg(fs[s]) for s in SINGLES}
    union = set().union(*cs.values())
    pres = lambda f: ({n for n in f if n not in fb}, {n for n in fb if n not in f})
    pd = pres(fd)
    ps_add = set().union(*(pres(fs[s])[0] for s in SINGLES))
    ps_rem = set().union(*(pres(fs[s])[1] for s in SINGLES))
    loops = {}
    for n in sorted(n for n in cd | union if LOOP.match(n)):
        base = counts(fb.get(n))
        deltas = {}
        for s in SINGLES:
            c = counts(fs[s].get(n))
            if base and c:
                deltas[s] = tuple(x - y for x, y in zip(c, base))
        pred = tuple(base[i] + sum(v[i] for v in deltas.values()) for i in range(3)) if base else None
        loops[n] = (base, deltas, pred, counts(fd.get(n)))
    return k, {"only_drop": cd - union, "only_singles": union - cd, "add": (pd[0], ps_add), "rem": (pd[1], ps_rem),
               "loops": loops}


E = {name: ac.elfs(name) for name in ["a_base", "a_drop5"] + ["a_" + s for s in SINGLES]}
keys = sorted(set.intersection(*(set(v) for v in E.values())))
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    res = [r for r in ex.map(one, keys) if r[1] is not None]
print(f"images {len(keys)}; differing from base in drop5 or a single: {len(res)}")
only_d = collections.Counter()
only_s = collections.Counter()
pres = collections.Counter()
loopsum = collections.defaultdict(lambda: collections.Counter())
mism = collections.defaultdict(collections.Counter)
for k, r in res:
    ce = k.split("__")[0] + " " + ("E1" if "subordinate" in k.split("__")[-1] else "E0")
    for n in r["only_drop"]:
        only_d[(ce, n)] += 1
    for n in r["only_singles"]:
        only_s[(ce, n)] += 1
    da, sa = r["add"]
    dr, sr = r["rem"]
    if da != sa or dr != sr:
        pres[(ce, f"drop5 adds {sorted(da)} removes {sorted(dr)}; singles add {sorted(sa)} remove {sorted(sr)}")] += 1
    for n, (base, deltas, pred, act) in r["loops"].items():
        fn = n.split("::", 1)[1]
        if pred == act:
            loopsum[(ce, fn)]["as predicted"] += 1
        else:
            loopsum[(ce, fn)]["differs"] += 1
            dl = " ".join(f"{s}{v[0]:+d}/{v[1]:+d}/{v[2]:+d}" for s, v in deltas.items() if any(v))
            mism[(ce, fn)][f"base {base} singles [{dl}] predicted {pred} drop5 {act}"] += 1
print("\n== functions changed in drop5 but in no single arm (config ERISC, function: images)")
for (ce, n), c in sorted(only_d.items()):
    print(f"   {ce:28s} {n}: {c}")
print("   (none)" if not only_d else "")
print("== functions changed by a single arm but not in drop5")
for (ce, n), c in sorted(only_s.items()):
    print(f"   {ce:28s} {n}: {c}")
print("   (none)" if not only_s else "")
print("== function presence (inlining) in drop5 not predicted by the singles")
for (ce, d), c in sorted(pres.items()):
    print(f"   {ce:28s} {d}: {c}")
print("   (none)" if not pres else "")
print("== pass loops: drop5's (instructions, stack loads, stack stores) vs base + the singles' summed deltas")
for (ce, fn), c in sorted(loopsum.items()):
    line = f"   {ce:28s} {fn:22s} as predicted {c['as predicted']:4d}, differs {c['differs']:4d}"
    print(line)
    for m, n in mism[(ce, fn)].most_common(3):
        print(f"      x{n}: {m}")
