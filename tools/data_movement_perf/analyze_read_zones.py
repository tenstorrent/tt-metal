"""Pair device zone markers per core/run and cross-check the actual Tracy export.

No timestamps are subtracted across cores. First two launches are correctness/warmup;
the last three are measured trials in this benchmark's capture protocol.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(folder, cores):
    path = folder / f"c{cores}_zones_tracy_device.csv"
    with path.open() as f:
        clock = int(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", next(f)).group(1))
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in reader]
    stacks = defaultdict(list)
    zones = defaultdict(dict)
    marker_counts = defaultdict(int)
    for r in rows:
        if r["RISC processor type"] != "NCRISC":
            continue
        key = (int(r["run host ID"]), int(r["core_x"]), int(r["core_y"]))
        name = r["zone name"]
        if name not in ("NCRISC-KERNEL", "DM_READ_TOTAL", "DM_READ_ISSUE", "DM_READ_BARRIER"):
            continue
        stamp = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            stacks[key, name].append(stamp)
        elif r["type"] == "ZONE_END":
            assert stacks[key, name], (key, name, "orphan end")
            start = stacks[key, name].pop()
            assert name not in zones[key], "This analysis expects kernel-iters=1"
            zones[key][name] = [start, stamp]
        marker_counts[key] += 1
    assert not any(stacks.values()), "Unclosed zones"
    groups = defaultdict(list)
    all_durations = defaultdict(list)
    for key, z in zones.items():
        if "DM_READ_TOTAL" not in z:
            continue
        assert len(z) == 4, (key, z)
        run, x, y = key
        origin, finish = z["NCRISC-KERNEL"]
        assert (
            origin
            <= z["DM_READ_TOTAL"][0]
            <= z["DM_READ_ISSUE"][0]
            < z["DM_READ_ISSUE"][1]
            <= z["DM_READ_BARRIER"][0]
            < z["DM_READ_BARRIER"][1]
            <= z["DM_READ_TOTAL"][1]
            <= finish
        )
        item = dict(
            run=run,
            core=[x, y],
            kernel_us=(finish - origin) / clock,
            spans_us={name: [(a - origin) / clock, (b - origin) / clock] for name, (a, b) in z.items()},
            durations_us={name: (b - a) / clock for name, (a, b) in z.items()},
        )
        groups[run].append(item)
        for name, (a, b) in z.items():
            if name.startswith("DM_READ"):
                all_durations[name].append((b - a) * 1000 / clock)
    # Verify events exported from the .tracy file agree with the device CSV.
    exported = defaultdict(list)
    with (folder / f"c{cores}_tracy_zones.csv").open() as f:
        for r in csv.DictReader(f):
            exported[r["name"]].append(float(r["GPU execution time"]))
    for name, durations in all_durations.items():
        assert len(durations) == len(exported[name]), (name, len(durations), len(exported[name]))
        assert all(abs(a - b) < 3 for a, b in zip(sorted(durations), sorted(exported[name]))), name
    run_ids = sorted(groups)
    report = json.loads((folder / f"c{cores}_zones_tracy.json").read_text())
    assert len(run_ids) == report["trials"] + 2
    for items in groups.values():
        assert len(items) == cores
    samples = report["results"][0]["samples_us"]
    median_index = sorted(range(len(samples)), key=lambda i: samples[i])[len(samples) // 2]
    selected_run = run_ids[median_index + 2]
    selected = groups[selected_run]
    critical = max(selected, key=lambda r: r["kernel_us"])
    return dict(
        cores=cores,
        clock_mhz=clock,
        device_program_us=samples[median_index],
        selected_run=selected_run,
        critical_core=critical,
        per_core=selected,
        max_to_median_kernel=max(r["kernel_us"] for r in selected)
        / statistics.median(r["kernel_us"] for r in selected),
        max_instrumented_markers=max(marker_counts.values()),
        tracy_zone_events_verified=sum(map(len, exported.values())),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, default=Path("generated/dram_l1_zones"))
    args = p.parse_args()
    data = [analyze(args.folder, n) for n in (7, 44)]
    (args.folder / "analysis.json").write_text(json.dumps(data, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
    colors = {"DM_READ_ISSUE": "#2878b5", "DM_READ_BARRIER": "#e58b2b"}
    for ax, d in zip(axs, data):
        c = d["critical_core"]
        ax.broken_barh([(0, c["kernel_us"])], (-0.30, 0.60), facecolors="#d5d9df")
        for name, color in colors.items():
            start, end = c["spans_us"][name]
            ax.broken_barh([(start, end - start)], (-0.25, 0.50), facecolors=color)
            ax.text(
                (start + end) / 2,
                0,
                f"{name.removeprefix('DM_READ_').lower()}\n{end-start:.3f} µs",
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
        ax.axvline(16.384, color="#b63036", linestyle="--", label="16.384 µs ideal whole-transfer DRAM floor")
        ax.set_title(
            f"{d['cores']} readers: longest-duration NCRISC core {tuple(c['core'])}, selected median trial; local kernel {c['kernel_us']:.3f} µs",
            loc="left",
            fontsize=11,
        )
        ax.set_xlim(0, 29)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xlabel("Time from this core’s NCRISC kernel entry (µs)")
        ax.grid(axis="x", alpha=0.2)
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Measured per-core timeline — BH, 1350 MHz, 8 MiB, 2 KiB reads, NoC0", fontsize=13)
    fig.savefig(args.folder / "one_core_timeline.png", dpi=180)
    fig.savefig(args.folder / "one_core_timeline.svg")
    for d in data:
        print(json.dumps({k: v for k, v in d.items() if k != "per_core"}, indent=2))


if __name__ == "__main__":
    main()
