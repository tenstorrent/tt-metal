"""cmp_probe_saves.py A B : bit-compare two PROBE_CMP_SAVE files (tests/test_chunk_probe.py, PROBE_CMP=1 PROBE_APC_GB=..) of two runs, e.g. DSV41_PF_POS_DEV=0 / 1: first tokens, prefill logits, decode logits,
decode index keys, prefix snapshots (carried state + pool rows) of every rep. Exit 1 on any difference."""
import re
import sys

import torch

a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
bad = 0


def stale(name):  # halo of a layer >= 21 in a snapshot: unspecified under the bounded replay (differs between two identical runs)
    m = re.match(r"L(\d+)\.halo", name)
    return bool(m) and int(m.group(1)) >= 21


def eq(name, x, y):
    global bad
    ok = x == y if not isinstance(x, torch.Tensor) else (x.shape == y.shape and torch.equal(x, y))
    if not ok:
        bad += 1
    return ok


for rep in a["cmp"]:
    ca, cb = a["cmp"][rep], b["cmp"][rep]
    r = {
        "first tokens": eq("tok", ca[0], cb[0]),
        "decode logits": all(eq("dec", x, y) for x, y in zip(ca[1], cb[1])) and len(ca[1]) == len(cb[1]),
        "decode keys": ca[2].keys() == cb[2].keys() and all(eq("key", ca[2][k], cb[2][k]) for k in ca[2]),
        "prefill logits": (ca[3] is None and cb[3] is None) or eq("lg", ca[3], cb[3]),
    }
    sa, sb = a["snaps"][rep], b["snaps"][rep]
    ok = sa.keys() == sb.keys()
    if ok:
        for k in sa:
            ok = ok and all(
                eq("carried", x, y) for n in sa[k][0] if not stale(n) for x, y in zip(sa[k][0][n], sb[k][0][n])
            )
            ok = ok and all(eq("pool", sa[k][1][n], sb[k][1][n]) for n in sa[k][1])
    r[f"{len(sa)}/{len(sb)} snapshots"] = bool(ok) and not bad
    print(f"rep {rep}: " + ", ".join(f"{k} {'EQUAL' if v else 'DIFFER'}" for k, v in r.items()), flush=True)
    bad += sum(not v for v in r.values()) - 0
print("RESULT", "BIT-IDENTICAL" if bad == 0 else f"MISMATCH ({bad})")
sys.exit(1 if bad else 0)
