# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Compare two pytest_device_time.py outputs (e.g. matmul_auto_config_v2 off vs on).

  python compare_pytest_times.py off.jsonl on.jsonl [--min-us 20] [--auto-only] [--write-auto-list FILE]

Reports outcome changes (a test passing in one run and failing in the other), and for tests that passed in
both, the change in total device-kernel time. Tests whose device time is below --min-us in both runs are left
out of the timing comparison as too noisy.

--auto-only compares just the tests in which some matmul went through the default config selection (auto_config
set in either run); the others pass their own program configs, so the flag cannot change them.
--write-auto-list FILE writes those tests' node ids to FILE (the pytest-auto suite's test list).
"""

import argparse
import json
import math
from collections import Counter


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["test"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--min-us", type=float, default=20.0)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--auto-only", action="store_true")
    ap.add_argument("--write-auto-list", metavar="FILE")
    args = ap.parse_args()
    base, new = load(args.base), load(args.new)
    common = [t for t in base if t in new]
    auto = [t for t in common if base[t].get("auto_config") or new[t].get("auto_config")]
    if args.write_auto_list:
        with open(args.write_auto_list, "w") as f:
            f.write("".join(t + "\n" for t in sorted(auto)))
    print(
        f"{len(common)} tests in both ({len(base)} base, {len(new)} new), {len(auto)} use the default config selection"
    )
    if args.auto_only:
        common = auto
        print(f"comparing those {len(auto)} only")
    print("outcomes base:", dict(Counter(base[t]["outcome"] for t in common)))
    print("outcomes new: ", dict(Counter(new[t]["outcome"] for t in common)))

    changed = [(t, base[t]["outcome"], new[t]["outcome"]) for t in common if base[t]["outcome"] != new[t]["outcome"]]
    print(f"\n{len(changed)} outcome changes:")
    for t, b, n in changed:
        print(f"  {b:>7s} -> {n:7s} {t}")

    ratios = []
    for t in common:
        b, n = base[t], new[t]
        if b["outcome"] != "passed" or n["outcome"] != "passed" or not b.get("device_ns") or not n.get("device_ns"):
            continue
        if b["device_ns"] < args.min_us * 1e3 and n["device_ns"] < args.min_us * 1e3:
            continue
        ratios.append((b["device_ns"] / n["device_ns"], t, b, n))
    if not ratios:
        return
    geo = math.exp(sum(math.log(r[0]) for r in ratios) / len(ratios))
    print(
        f"\ndevice time (base/new, >= {args.min_us:g} us): {len(ratios)} tests, geomean {geo:.3f}, "
        f"faster >5%: {sum(r[0] > 1.05 for r in ratios)}, slower >5%: {sum(r[0] < 0.95 for r in ratios)}, "
        f"within 5%: {sum(0.95 <= r[0] <= 1.05 for r in ratios)}"
    )
    ratios.sort()
    print("\nLargest slowdowns:")
    for r, t, b, n in ratios[: args.show]:
        if r >= 0.95:
            break
        print(
            f"  {r:5.2f}x {b['device_ns'] / 1e3:9.1f} -> {n['device_ns'] / 1e3:9.1f}us ({b['programs']}->{n['programs']}p) {t}"
        )
    print("\nLargest speedups:")
    for r, t, b, n in reversed(ratios[-args.show :]):
        if r <= 1.05:
            break
        print(
            f"  {r:5.2f}x {b['device_ns'] / 1e3:9.1f} -> {n['device_ns'] / 1e3:9.1f}us ({b['programs']}->{n['programs']}p) {t}"
        )


if __name__ == "__main__":
    main()
