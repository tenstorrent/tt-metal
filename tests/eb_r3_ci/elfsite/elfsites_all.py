#!/usr/bin/env python3
"""For every build that differs between two JIT caches (elfcmp.py keys), attribute the difference to call sites (elfsite.py)
and aggregate: per site, in how many builds its Tensix instructions changed, with the change of each Tensix mnemonic; sites whose
Tensix instructions are unchanged (only the RISC-V instruction count moved) are counted apart.
usage: elfsites_all.py <cache A> <cache B> <kernel regex> [--shift=N]"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elfcmp  # noqa: E402
import elfsite  # noqa: E402


def main():
    a, b, kre = sys.argv[1], sys.argv[2], sys.argv[3]
    shift = next((int(x.split("=")[1]) for x in sys.argv[4:] if x.startswith("--shift=")), 0)
    da, pa = elfcmp.load(a, None, kre)
    db, pb = elfcmp.load(b, None, kre)
    tt_changed = collections.defaultdict(lambda: [0, collections.Counter()])
    count_only = collections.Counter()
    nbuilds = collections.Counter()
    for k in sorted(set(da) & set(db)):
        if da[k] == db[k]:
            continue
        ea = [pa[(k, h)] for h in da[k]]
        eb = [pb[(k, h)] for h in db[k]]
        nbuilds[k[2]] += 1
        for x, y in zip(sorted(ea), sorted(eb)):
            for site, ca, cb, ta, tb in elfsite.compare(x, y, shift=shift):
                key = (k[2], site)
                if ta != tb:
                    tt_changed[key][0] += 1
                    for m in set(ta) | set(tb):
                        tt_changed[key][1][m.split()[0]] += tb[m] - ta[m]
                else:
                    count_only[key] += 1
    print("differing builds per thread:", dict(nbuilds))
    print("sites whose Tensix instructions changed:")
    for (th, site), (n, delta) in sorted(tt_changed.items()):
        d = ", ".join(f"{m} {v:+d}" for m, v in sorted(delta.items()) if v)
        print(f"  {th} {site}: {n} builds; net per-mnemonic change over those builds: {d or 'operands only'}")
    print("sites with only a RISC-V instruction-count change:")
    for (th, site), n in sorted(count_only.items()):
        print(f"  {th} {site}: {n} builds")


if __name__ == "__main__":
    main()
