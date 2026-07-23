# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct and plot Quasar cache-write results (test id 912) directly from the
profiler device CSV, without relying on ``run_host_id``.

TEMPORARY: this is a stopgap script. Its CSV-order reconstruction is meant to be
folded into the shared dm pipeline (``stats_collector.aggregate_performance``) as
an order-based fallback for when ``run_host_id`` collides, at which point
``test_data_movement.py --plot`` plots these curves directly and this standalone
script should be removed.

Why this exists
---------------
On the ``emu-quasar-1x3`` target there are no fast-dispatch cores, so the test
runs in slow-dispatch mode. Under slow dispatch the profiler does not increment
``run host ID`` -- every sweep run is stamped ``run_host_id = 0``. The shared dm
harness (``stats_collector.aggregate_performance``) groups runs by
``run_host_id``, so all runs collapse into a single point and the normal
``--plot`` output is unusable.

The per-run data is still fully present in the CSV, in execution order: each
``RISCV1`` ``ZONE_START``/``ZONE_END`` pair is immediately followed by that run's
``Transaction size in bytes`` and ``Write path`` stamps. This script walks the
CSV in order to recover each run's (size, mode, duration) tuple and plots all
three write modes overlaid, as both duration (cycles) and bandwidth
(bytes/cycle), with a zoomed panel over the small-size crossover region.

Usage
-----
    python tests/tt_metal/tt_metal/data_movement/quasar_cache_perf/plot_cache_write_from_csv.py \
        [--csv generated/profiler/.logs/profile_log_device.csv] \
        [--out "tests/tt_metal/tt_metal/data_movement/data/quasar/Quasar Cache Write Sizes.png"] \
        [--zoom-max 64]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TEST_ID = 912
# Write modes stamped as "Write path" by the kernel.
MODE_LABELS = {0: "Uncached (1B)", 1: "Uncached (8B)", 2: "Cached+Flush (8B)"}

# CSV column indices (0-based), per the tracy device-log header:
# PCIe, core_x, core_y, RISC type, timer_id, time, data, run host ID, trace id,
# trace id counter, zone name, type, source line, source file, meta
COL_TIME = 5
COL_DATA = 6
COL_ZONE = 10
COL_TYPE = 11


def reconstruct_runs(csv_path):
    """Return a list of {'dur','size','mode','test_id'} dicts, one per RISCV1 run,
    recovered from CSV order (independent of run_host_id)."""
    with open(csv_path) as f:
        rows = [line.rstrip("\n").split(",") for line in f]
    runs = []
    cur = {}
    start = None
    for r in rows:
        if len(r) <= COL_TYPE:
            continue
        zone, typ = r[COL_ZONE], r[COL_TYPE]
        if typ == "ZONE_START" and zone == "RISCV1":
            start = int(r[COL_TIME])
        elif typ == "ZONE_END" and zone == "RISCV1" and start is not None:
            cur = {"dur": int(r[COL_TIME]) - start}
            start = None
        elif typ == "TS_DATA" and cur:
            val = int(r[COL_DATA])
            if zone == "Transaction size in bytes":
                cur["size"] = val
            elif zone == "Write path":
                cur["mode"] = val
            elif zone == "Test id":
                cur["test_id"] = val
            # "Write path" is stamped last per run -> flush the run.
            if zone == "Write path":
                runs.append(cur)
                cur = {}
    return [r for r in runs if r.get("test_id", TEST_ID) == TEST_ID and "size" in r and "mode" in r]


def series_by_mode(runs):
    """{mode: {'sizes':[...], 'dur':[...], 'bw':[...]}} for each write mode present."""
    by_mode = {}
    for r in runs:
        by_mode.setdefault(r["mode"], {})[r["size"]] = r["dur"]
    out = {}
    for mode, size_to_dur in by_mode.items():
        sizes = sorted(size_to_dur)
        durs = [size_to_dur[s] for s in sizes]
        bw = [s / d if d else 0 for s, d in zip(sizes, durs)]
        out[mode] = {"sizes": sizes, "dur": durs, "bw": bw}
    return out


def _draw(ax, series, ykey, size_filter, title, ylabel):
    for mode in sorted(series):
        s = series[mode]
        pts = [(x, y) for x, y in zip(s["sizes"], s[ykey]) if size_filter(x)]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=MODE_LABELS.get(mode, f"mode {mode}"))
        ax.set_xticks(xs)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Total Data Size (bytes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument(
        "--out", default="tests/tt_metal/tt_metal/data_movement/data/quasar/Quasar Cache Write Sizes.png"
    )
    parser.add_argument("--zoom-max", type=int, default=64, help="upper size (bytes) for the zoomed panels")
    args = parser.parse_args()

    runs = reconstruct_runs(args.csv)
    if not runs:
        raise SystemExit(f"No test {TEST_ID} runs reconstructed from {args.csv}")
    series = series_by_mode(runs)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    _draw(axes[0][0], series, "dur", lambda s: True, "Duration - full range", "Duration (cycles)")
    _draw(
        axes[0][1],
        series,
        "dur",
        lambda s: s <= args.zoom_max,
        f"Duration - zoom <= {args.zoom_max}B",
        "Duration (cycles)",
    )
    _draw(axes[1][0], series, "bw", lambda s: True, "Bandwidth - full range", "Bandwidth (bytes/cycle)")
    _draw(
        axes[1][1],
        series,
        "bw",
        lambda s: s <= args.zoom_max,
        f"Bandwidth - zoom <= {args.zoom_max}B",
        "Bandwidth (bytes/cycle)",
    )
    fig.suptitle(
        f"Quasar Cache Write Sizes (test {TEST_ID}) - uncached 1B vs uncached 8B vs cached+flush", fontweight="bold"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Reconstructed {len(runs)} runs. Saved plot: {args.out}")


if __name__ == "__main__":
    main()
