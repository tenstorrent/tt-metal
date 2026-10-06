# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Read sweep results (the sweep_enumerated.py format) and report how close each selection comes to the best config.

A sweep holds, per problem (matmul call), many timed configs. A selection's regret on a problem is its config's time
over the fastest config timed for that problem (1.0: it picked the best). The report gives, per origin (the v2
heuristic, the legacy selection), the geomean and worst regret over the problems, and the worst problems.

  python sweep_data.py sweep_wh.csv [more.csv ...] [--min-us 20] [--worst 10]

Problems whose best config is faster than --min-us are left out: run-to-run noise on tiny kernels (10-30%) would
dominate their regret.
"""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Timed:
    config: str
    origin: str
    family: str
    device_ns: float
    device_ns_min: float
    device_ns_max: float
    fields: dict


@dataclass
class Problem:
    problem_id: str
    case: str
    fields: dict
    timed: list = field(default_factory=list)  # Timed, status ok
    failed: list = field(default_factory=list)  # rows of configs that didn't run or failed PCC

    def best(self):
        return min(self.timed, key=lambda t: t.device_ns) if self.timed else None

    def of(self, origin):
        """The timed config of `origin` (the first one), or None"""
        return next((t for t in self.timed if t.origin == origin), None)

    def regret(self, origin):
        chosen, best = self.of(origin), self.best()
        return chosen.device_ns / best.device_ns if chosen and best else None


def _float(v):
    return float(v) if v not in ("", None) else math.nan


def load(paths):
    """The problems of one or more sweep CSVs, by problem_id. A (problem, origin, config) row that appears again (a
    later run, or a later file) replaces the earlier one."""
    latest = {}
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                latest[(r["problem_id"], r["origin"], r["config"])] = r
    problems = {}
    for r in latest.values():
        p = problems.setdefault(r["problem_id"], Problem(r["problem_id"], r["case"], r))
        if r["status"] == "ok" and r.get("device_ns"):
            p.timed.append(
                Timed(
                    config=r["config"],
                    origin=r["origin"],
                    family=r["family"],
                    device_ns=_float(r["device_ns"]),
                    device_ns_min=_float(r.get("device_ns_min")),
                    device_ns_max=_float(r.get("device_ns_max")),
                    fields=r,
                )
            )
        else:
            p.failed.append(r)
    return problems


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else math.nan


def report(problems, min_us=20.0, worst=10, origins=("heuristic", "legacy")):
    kept = [p for p in problems.values() if p.best() and p.best().device_ns >= min_us * 1e3]
    print(f"{len(problems)} problems, {len(kept)} with a best config of {min_us:g} us or more")
    statuses = defaultdict(int)
    for p in problems.values():
        statuses["ok"] += len(p.timed)
        for r in p.failed:
            statuses[r["status"]] += 1
    print("configs by status:", dict(statuses))
    for origin in origins:
        regrets = [(p.regret(origin), p) for p in kept if p.regret(origin) is not None]
        if not regrets:
            continue
        values = [r for r, _ in regrets]
        at_best = sum(1 for r in values if r <= 1.03)
        print(
            f"\n{origin}: {len(values)} problems, geomean regret {geomean(values):.3f}, worst {max(values):.2f}, "
            f"within 3% of the best on {at_best}"
        )
        for r, p in sorted(regrets, key=lambda x: -x[0])[:worst]:
            best = p.best()
            print(
                f"  {r:5.2f}x  {p.case:44s} {p.of(origin).device_ns / 1e3:9.1f} us vs best {best.device_ns / 1e3:9.1f} us "
                f"({best.family}, k={best.fields['in0_block_w']}, block={best.fields['out_block_h']}x"
                f"{best.fields['out_block_w']}, per_core={best.fields['per_core_M']}x{best.fields['per_core_N']})"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--min-us", type=float, default=20.0)
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args()
    report(load(args.csv), args.min_us, args.worst)


if __name__ == "__main__":
    main()
