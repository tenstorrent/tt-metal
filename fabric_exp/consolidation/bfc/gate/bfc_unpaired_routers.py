#!/usr/bin/env python3
"""bfc_unpaired_routers.py BASE_COV NEW_COV CASE...: router builds bfc_gate.py could not pair (the multi-rank logs
lose or garble some compile lines; the streaming profiler numbers its translation units per run). Each is paired here
by the case's JIT cache instead of the log: same processor, same named args (the new arg stripped), and, for the
streaming cases, the same compile flags except -DTT_PROFILER_TU_ID. Prints paired identical / differ / still unpaired."""
import collections, glob, os, re, sys
import importlib.util
spec = importlib.util.spec_from_file_location("g", os.path.dirname(os.path.abspath(__file__)) + "/bfc_gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

def cache_index(cov, case):
    idx = collections.defaultdict(list)
    for nf in glob.glob(os.path.join(cov, "cases", case, "cache", "*", "kernels", "fabric_erisc_router", "*", "named_ct_arg_map_generated.h")):
        named = re.sub(r',?\{"' + g.ARG + r'",\d+\}', "", open(nf).read())
        for proc in ("active_erisc", "subordinate_active_erisc"):
            elf = os.path.join(os.path.dirname(nf), proc, proc + ".elf")
            if os.path.exists(elf):
                idx[(proc, named)].append(elf)
    return idx

base_cov, new_cov = sys.argv[1], sys.argv[2]
tot = collections.Counter()
for case in sys.argv[3:]:
    a, b = g.builds(base_cov, case), g.builds(new_cov, case)
    ia, ib = cache_index(base_cov, case), cache_index(new_cov, case)
    per = collections.Counter()
    for side, mine, other_idx, other_logged in (("base", a, ib, b), ("new", b, ia, a)):
        for k, v in mine.items():
            if k in other_logged or k[0] != "fabric_erisc_router":
                continue
            cands = other_idx.get((k[1], k[3]), [])
            if not cands:
                per[f"{side} still unpaired"] += 1
                continue
            mine_img = g.cov.image(v[0][1])
            same = any(g.cov.image(c) == mine_img for c in cands)
            per[f"{side} paired by cache: {'identical' if same else 'differ'}"] += 1
    tot.update(per)
    print(f"{case:26s} " + "  ".join(f"{x} {per[x]}" for x in sorted(per)))
print("totals: " + "  ".join(f"{x} {tot[x]}" for x in sorted(tot)))
