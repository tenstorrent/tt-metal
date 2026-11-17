#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Idle drift of the DRAM tile NIU counters with nothing running, and who causes it. Read-only.

Run one mode per process (each leaves a json), then `compare`:

  python dram_idle_drift.py run --mode umd            --out umd.json
  python dram_idle_drift.py run --mode exalens        --out exalens.json
  python dram_idle_drift.py run --mode exalens-reinit --out reinit.json
  python dram_idle_drift.py compare umd.json exalens.json reinit.json

For every mode the run takes snapshots at several spacings and fits, per counter group, words = a + b * seconds:
  a (words per snapshot pair) is the cost of reading, b (words per second) is traffic of something else.
It also reads one register many times back to back to see if a read itself increments a counter.
Reading the sums of read words, not per tile, keeps the output short; the json has every tile.
"""

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _common as c

SPACINGS_S = (0.0, 0.0, 0.05, 0.05, 0.5, 0.5, 2.0, 2.0, 5.0)
SELF_READS = 200


def fit(xs, ys):
    """Least squares y = a + b x; (a, b)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def run(args):
    if c.card_busy(args.device):
        c.fail(f"{args.device} is open by another process; not reading")
    reader = c.make_reader(args.mode, args.device)
    tiles = reader.tiles()
    samples = []
    for spacing in SPACINGS_S:
        before = c.snapshot(reader, tiles, args.arch)
        t0 = time.monotonic()
        if spacing:
            time.sleep(spacing)
        after = c.snapshot(reader, tiles, args.arch)
        elapsed = time.monotonic() - t0
        samples.append(
            {
                "spacing_s": spacing,
                "elapsed_s": elapsed,
                "delta": {
                    "|".join(map(str, k[:2]))
                    + "|"
                    + k[2]: c.wrap_delta(after[k], before[k])
                    for k in before
                },
            }
        )
    probe_tile = tiles[0]
    addr = c.niu_addr(probe_tile, 0, c.NIU_COUNTERS["RD_DATA_WORD_SENT"], args.arch)
    values = [reader.read32(probe_tile, addr) for _ in range(SELF_READS)]
    json.dump(
        {
            "mode": args.mode,
            "tiles": tiles,
            "samples": samples,
            "self_reads": {
                "tile": probe_tile,
                "values": values,
                "per_read": c.wrap_delta(values[-1], values[0]) / (SELF_READS - 1),
            },
        },
        open(args.out, "w"),
        indent=1,
    )
    print(f"{args.mode}: {len(samples)} samples, {len(tiles)} tiles -> {args.out}")


def group_total(sample, suffix):
    return sum(v for k, v in sample["delta"].items() if k.endswith("|" + suffix))


def compare(args):
    for path in args.files:
        d = json.load(open(path))
        print(f"== {d['mode']}  ({len(d['tiles'])} DRAM tiles)")
        xs = [s["elapsed_s"] for s in d["samples"]]
        for suffix in (
            "RD_DATA_WORD_SENT",
            "NONPOSTED_WR_DATA_WORD_RECEIVED",
            "RD_REQ_RECEIVED",
        ):
            ys = [group_total(s, suffix) for s in d["samples"]]
            a, b = fit(xs, ys)
            print(
                f"  {suffix:34s} per pair {a:9.1f} words  per second {b:9.1f} words/s   "
                f"(min {min(ys)}, max {max(ys)} over {len(ys)} pairs)"
            )
        print(
            f"  back-to-back reads of one register: {d['self_reads']['per_read']:.3f} words per read"
        )
    print(
        "\nReading guide: a rate b that is the same in every mode is traffic of something other than the reader "
        "(ARC, host, device). A per-pair cost a or a per-read increment that appears in one mode only is that "
        "reader. If a and b are near zero in the umd mode but not in the exalens modes, exalens causes it."
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--mode",
        required=True,
        choices=("umd", "exalens", "exalens-reinit", "exalens-unsafe"),
    )
    r.add_argument("--out", required=True)
    r.add_argument("--arch", default="blackhole", choices=("blackhole", "wormhole"))
    r.add_argument("--device", default=c.default_device_path())
    r.set_defaults(fn=run)
    cp = sub.add_parser("compare")
    cp.add_argument("files", nargs="+")
    cp.set_defaults(fn=compare)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
