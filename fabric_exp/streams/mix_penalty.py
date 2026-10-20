#!/usr/bin/env python3
"""Mix penalty for the P3 stream mixes.

usage: mix_penalty.py BUILD=CSV_PREFIX ...   (CSV_PREFIX is a results path without .csv/.log)

For each location and mix: measured cycles/packet, the prediction from pure runs in the same session
(sum over segments of share x pure cycles/packet), their ratio (the mix penalty), and the share of
packets that had to copy a header into a router slot. Cycles/packet is 1.35 GHz / packets per second
from the summary row, the same as the study's cyc.py; for the forwarded case that is the shared link.
"""
import csv
import re
import sys

CLOCK_HZ = 1.35e9
LOCATIONS = ["EW", "NS", "FWD"]
# mix -> [(pure test, share of packets per lap)]
MIXES = {
    "MixCmd": [("PureWrite4352", 1 / 4), ("PureScatter4352", 1 / 4), ("PureFused4352", 1 / 4), ("PureAtomic", 1 / 4)],
    "MixSize2": [("PureWrite4352", 1 / 2), ("PureWrite16", 1 / 2)],
    "MixSize4": [("PureWrite4352", 1 / 4), ("PureWrite1024", 1 / 4), ("PureWrite256", 1 / 4), ("PureWrite16", 1 / 4)],
    "MixChunk": [("PureWrite4352", 7 / 8), ("PureFused4352", 1 / 8)],
}
PURES = [
    "PureWrite4352",
    "PureScatter4352",
    "PureFused4352",
    "PureAtomic",
    "PureWrite1024",
    "PureWrite256",
    "PureWrite16",
]


def load(prefix):
    rows = {r["test_name"]: r for r in csv.DictReader(open(prefix + ".csv"))}
    cycles = {name: CLOCK_HZ / float(r["avg_packets_per_s"]) for name, r in rows.items()}
    gbs = {name: float(r["avg_bandwidth_gigabytes_per_s"]) for name, r in rows.items()}
    copies = {}
    test = None
    for line in open(prefix + ".log"):
        m = re.search(r"Running Test: (\S+)", line)
        if m:
            # Trimmed runs: results come from the replay pass, so ignore the capture pass's counters.
            test = m.group(1)
            if test.endswith("_capture"):
                test = None
            elif test.endswith("_replay"):
                test = test[: -len("_replay")]
        m = re.search(r": (\d+) of (\d+) packets (?:copied a header|wrote a header into their router slot)", line)
        if m and test:
            c, p = copies.get(test, (0, 0))
            copies[test] = (c + int(m.group(1)), p + int(m.group(2)))
    worker_bound = sum(1 for line in open(prefix + ".log") if "was worker-bound" in line)
    return cycles, gbs, copies, worker_bound


def main():
    builds = [arg.split("=", 1) for arg in sys.argv[1:]]
    data = {name: load(prefix) for name, prefix in builds}

    print("Worker-bound flags per build: " + ", ".join(f"{b} {data[b][3]}" for b, _ in builds) + "\n")
    print("## Pure runs, cycles/packet\n")
    print("| Location | Pure run | " + " | ".join(name for name, _ in builds) + " |")
    print("|---|---|" + "---:|" * len(builds))
    for loc in LOCATIONS:
        for pure in PURES:
            cells = [f"{data[b][0].get(f'{loc}_{pure}', float('nan')):.1f}" for b, _ in builds]
            print(f"| {loc} | {pure} | " + " | ".join(cells) + " |")

    print("\n## Mixes: measured / predicted cycles/packet, penalty, header copies\n")
    print("| Location | Mix | " + " | ".join(name for name, _ in builds) + " |")
    print("|---|---|" + "---|" * len(builds))
    for loc in LOCATIONS:
        for mix, parts in MIXES.items():
            cells = []
            for b, _ in builds:
                cycles, _, copies, _ = data[b]
                measured = cycles.get(f"{loc}_{mix}")
                predicted = sum(share * cycles.get(f"{loc}_{pure}", float("nan")) for pure, share in parts)
                if measured is None:
                    cells.append("-")
                    continue
                c, p = copies.get(f"{loc}_{mix}", (0, 0))
                copy_pct = 100.0 * c / p if p else 0.0
                cells.append(
                    f"{measured:.1f} / {predicted:.1f} = **{measured / predicted:.3f}** ({copy_pct:.0f}% copy)"
                )
            print(f"| {loc} | {mix} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
