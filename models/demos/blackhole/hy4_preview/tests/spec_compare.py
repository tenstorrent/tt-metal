"""spec_compare.py PLAIN.json SPEC.json: greedy-exactness gate (D5-1 gate 1): per user the first index where the spec stream differs from the plain stream and the plain top-1/top-2 logit gap there
(pass: no divergence, or gap < 0.05 at the first divergence). Plain json from run_spec_decode.py K=0 (gaps[r][s] = gap of the step that produced generated[r][s+1])."""
import json
import sys

plain, spec = (json.load(open(f)) for f in sys.argv[1:3])
ok = True
for r, (a, b) in enumerate(zip(plain["generated"], spec["generated"])):
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    if d is None:
        print(f"user {r}: identical over {n} tokens")
    else:
        gap = plain["gaps"][r][d - 1] if d >= 1 else None
        good = gap is not None and gap < 0.05
        ok &= good
        print(
            f"user {r}: first divergence at token {d} of {n}; plain top1-top2 gap {gap} -> {'near-tie (accepted)' if good else 'REAL DIVERGENCE'}"
        )
print("GREEDY_EXACTNESS", "PASS" if ok else "FAIL")
