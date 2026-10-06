#!/usr/bin/env python3
"""Summarise spec_adapt logs: python tools/spec_adapt_report.py <log> [...]  -> per-policy table (SPEC_RESULT_CYCLE / SPEC_RESULT lines) + confidence-head reliability (SPEC_CONF)."""
import json
import re
import sys

for f in sys.argv[1:]:
    print(f"== {f}")
    cyc, conf, ex = [], None, None
    for line in open(f, errors="replace"):
        if "SPEC_RESULT_CYCLE " in line:
            cyc.append(json.loads(line.split("SPEC_RESULT_CYCLE ", 1)[1]))
        elif "SPEC_CONF " in line:
            conf = json.loads(line.split("SPEC_CONF ", 1)[1])
        elif "SPEC exactness" in line:
            ex = re.search(r"(\d+)/(\d+) users identical", line)
    for r in cyc:
        print(
            f"  B={r['B']:3d} {r['policy']:6s} rounds {r['rounds']:3d} acc/round {r['accepted_per_round']:.2f} tok/round {r['tok_per_round']:.2f} round {r['round_ms']:6.1f} ms "
            f"-> {r['spec_tok_s_user']:5.1f} tok/s/user ({r['speedup']:.2f}x plain {r['plain_tok_s_user']:.1f}) k {r['k_hist']}"
        )
    if ex:
        print(f"  exact vs plain: {ex.group(1)}/{ex.group(2)}")
    if conf:
        print(
            "  confidence head, conditional P(draft j accepted | prefix ok): j: n pred obs | prefix survival pred obs"
        )
        for j in "12345":
            c, s = conf["cond"].get(j), conf["surv"].get(j)
            if c and s:
                print(
                    f"    j={j}: n={c['n']:4d} {c['pred']:.3f} {c['obs']:.3f} | n={s['n']:4d} {s['pred']:.3f} {s['obs']:.3f}"
                )
