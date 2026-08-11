# SPDX-License-Identifier: Apache-2.0
"""Compare two bench sides, each measured twice, and flag which deltas exceed the noise floor.

usage: compare_bench.py base1.json base2.json new1.json new2.json
"""
import json
import sys


def load(paths):
    runs = [json.load(open(p)) for p in paths]
    keys = set(runs[0])
    for r in runs[1:]:
        keys &= set(r)
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / mean * 100 if mean else 0.0
        out[k] = (mean, spread)
    return out


def main(argv):
    base = load(argv[0:2])
    new = load(argv[2:4])
    keys = sorted(set(base) & set(new))

    # Noise floor: the worst run-to-run spread seen on either side, over the untouched CTL cases.
    ctl_spreads = [max(base[k][1], new[k][1]) for k in keys if k.startswith("CTL.")]
    floor = max(ctl_spreads) if ctl_spreads else 0.0
    all_spreads = [max(base[k][1], new[k][1]) for k in keys]

    print(f"cases={len(keys)}  CTL noise floor={floor:.2f}%  worst spread over all cases={max(all_spreads):.2f}%\n")
    print(f"| case | baseline us | branch us | delta | spread b/n | verdict |")
    print(f"|---|---:|---:|---:|---:|---|")
    sig = []
    for k in keys:
        b, bs = base[k]
        n, ns = new[k]
        d = (n - b) / b * 100
        worst = max(bs, ns, floor)
        if abs(d) <= worst:
            verdict = "noise"
        elif d < 0:
            verdict = "**FASTER**"
            sig.append((k, d))
        else:
            verdict = "**SLOWER**"
            sig.append((k, d))
        print(f"| `{k}` | {b:.2f} | {n:.2f} | {d:+.2f}% | {bs:.1f}/{ns:.1f}% | {verdict} |")

    print()
    if sig:
        print("Outside the noise floor:")
        for k, d in sorted(sig, key=lambda x: x[1]):
            print(f"  {d:+7.2f}%  {k}")
    else:
        print("No case moved beyond the noise floor.")


if __name__ == "__main__":
    main(sys.argv[1:])
