"""Expand retained Tracy captures into every core/trial and firmware boundary duration."""

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "generated/dram_l1_zones"
OUT = ROOT / "ttnn/ttnn/operations/examples/dram_l1_transaction_size"
selected = {d["cores"]: d["selected_run"] for d in json.loads((FOLDER / "analysis.json").read_text())}
all_rows = []
programs = []
for count in (7, 44):
    with (FOLDER / f"c{count}_zones_tracy_device.csv").open() as f:
        clock = int(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", next(f)).group(1))
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f, skipinitialspace=True)]
    opened = {}
    spans = defaultdict(dict)
    for r in rows:
        name = r["zone name"]
        phase = r["type"]
        if phase not in ("ZONE_START", "ZONE_END"):
            continue
        key = (int(r["run host ID"]), int(r["core_x"]), int(r["core_y"]))
        zone = (r["RISC processor type"], name)
        stamp = int(r["time[cycles since reset]"])
        if phase == "ZONE_START":
            assert (key, zone) not in opened
            opened[key, zone] = stamp
        else:
            assert (key, zone) in opened
            assert zone not in spans[key]
            spans[key][zone] = (opened.pop((key, zone)), stamp)
    assert not opened
    runs = sorted({key[0] for key, z in spans.items() if ("NCRISC", "DM_READ_TOTAL") in z})
    with (FOLDER / f"c{count}_ops.csv").open() as f:
        ops = {int(r["GLOBAL CALL COUNT"]): r for r in csv.DictReader(f) if r["OP CODE"] == "GenericOpDeviceOperation"}
    for trial, run in enumerate(runs[2:], 1):
        op = ops[run]
        programs.append(
            dict(
                cores=count,
                trial=trial,
                run=run,
                selected=run == selected[count],
                host_call_us=float(op["HOST DURATION [ns]"]) / 1000,
                device_fw_us=float(op["DEVICE FW DURATION [ns]"]) / 1000,
                device_kernel_us=float(op["DEVICE KERNEL DURATION [ns]"]) / 1000,
                start_spread_us=float(op["DEVICE KERNEL FIRST TO LAST START [ns]"]) / 1000,
            )
        )
        for (rid, x, y), z in sorted(spans.items()):
            if rid != run or ("NCRISC", "DM_READ_TOTAL") not in z:
                continue
            fw = z["NCRISC", "NCRISC-FW"]
            k = z["NCRISC", "NCRISC-KERNEL"]
            total = z["NCRISC", "DM_READ_TOTAL"]
            issue = z["NCRISC", "DM_READ_ISSUE"]
            barrier = z["NCRISC", "DM_READ_BARRIER"]
            boundaries = [fw[0], k[0], total[0], issue[0], issue[1], barrier[0], barrier[1], total[1], k[1], fw[1]]
            assert boundaries == sorted(boundaries)
            names = [
                "fw_before_kernel",
                "kernel_before_total",
                "total_before_issue",
                "issue",
                "between_issue_barrier",
                "barrier",
                "total_after_barrier",
                "kernel_after_total",
                "fw_after_kernel",
            ]
            item = dict(cores=count, trial=trial, run=run, selected=run == selected[count], core_x=x, core_y=y)
            item.update({name + "_us": (b - a) / clock for name, a, b in zip(names, boundaries, boundaries[1:])})
            item["kernel_us"] = (k[1] - k[0]) / clock
            item["ncrisc_fw_us"] = (fw[1] - fw[0]) / clock
            br = z["BRISC", "BRISC-FW"]
            item["brisc_fw_us"] = (br[1] - br[0]) / clock
            item["other_kernel_us"] = item["kernel_us"] - item["issue_us"] - item["barrier_us"]
            assert abs(sum(item[name + "_us"] for name in names) - item["ncrisc_fw_us"]) < 1e-10
            all_rows.append(item)
assert len(all_rows) == (7 + 44) * 3
for name, data in [("all_core_durations.csv", all_rows), ("program_durations.csv", programs)]:
    with (OUT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={"width_ratios": [1, 1.4]}, layout="constrained")
for ax, n in zip(axs, (7, 44)):
    data = sorted([r for r in all_rows if r["cores"] == n and r["selected"]], key=lambda r: (r["core_y"], r["core_x"]))
    for i, r in enumerate(data):
        ax.barh(i, r["issue_us"], color="#2878b5", label="Issue (includes readiness waits)" if i == 0 else None)
        ax.barh(i, r["barrier_us"], left=r["issue_us"], color="#e58b2b", label="Final barrier" if i == 0 else None)
        ax.barh(
            i,
            r["other_kernel_us"],
            left=r["issue_us"] + r["barrier_us"],
            color="#adb3bd",
            label="Other kernel time" if i == 0 else None,
        )
    ax.set_yticks(range(len(data)), [f"({r['core_x']},{r['core_y']})" for r in data])
    ax.invert_yaxis()
    ax.set_xlim(0, 29)
    ax.set_xlabel("Local kernel duration (µs)")
    ax.set_ylabel("Profiler core coordinate")
    ax.set_title(f"{n} readers — selected median trial")
    ax.grid(axis="x", alpha=0.2)
    ax.axvline(16.384, ls="--", color="#b63036", lw=1)
axs[0].legend(loc="lower left", fontsize=9)
fig.suptitle(
    "All cores: read issue, remaining wait, other kernel time\nBH 1350 MHz · 8 MiB · 2 KiB transactions · NoC0\nDuration decomposition; bars do not imply synchronized core starts",
    fontsize=13,
)
fig.savefig(OUT / "all_core_durations.png", dpi=160)
fig.savefig(OUT / "all_core_durations.svg")

text = [
    "# All captured read durations",
    "",
    "Existing Tracy captures, 2026-10-08, BH at 1350 MHz; 8 MiB BF16, 2048-byte reads on preferred NoC0. No new device run. All 153 core executions in the three measured trials per configuration were checked. Correctness and warmup launches are excluded.",
    "",
    "![All core durations](all_core_durations.png)",
    "",
    "Bars show summed local durations, not a globally synchronized timeline. Each timestamp difference is within one core. Coordinates are those recorded by the profiler.",
    "",
    "## Whole-program measurements",
    "",
    "| Readers | Trial | Selected | Kernel µs | Firmware µs | First-to-last kernel start µs | Host call µs |",
    "|---:|---:|:---:|---:|---:|---:|---:|",
]
for r in programs:
    text.append(
        f"| {r['cores']} | {r['trial']} | {'yes' if r['selected'] else ''} | {r['device_kernel_us']:.3f} | {r['device_fw_us']:.3f} | {r['start_spread_us']:.3f} | {r['host_call_us']:.3f} |"
    )
text += [
    "",
    "Firmware and kernel spans overlap; do not add them. Host API-call duration is a separate host-side measurement, not transfer time or end-to-end latency. The start spread is the profiler-reported metric, not a subtraction of raw cross-core clocks.",
    "",
]
for n in (7, 44):
    data = [r for r in all_rows if r["cores"] == n and r["selected"]]
    text += [
        f"## {n} readers: every core in the selected median trial",
        "",
        "| Core | Issue µs | Barrier µs | Other kernel µs | Kernel µs | NCRISC firmware µs | BRISC firmware µs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(data, key=lambda r: (r["core_y"], r["core_x"])):
        text.append(
            f"| ({r['core_x']},{r['core_y']}) | "
            + " | ".join(
                f"{r[k]:.3f}"
                for k in ["issue_us", "barrier_us", "other_kernel_us", "kernel_us", "ncrisc_fw_us", "brisc_fw_us"]
            )
            + " |"
        )
    text += [
        "",
        f"### {n} readers: smallest boundary durations (all measured cores/trials)",
        "",
        "| Region | Min µs | Median µs | Max µs |",
        "|---|---:|---:|---:|",
    ]
    every = [r for r in all_rows if r["cores"] == n]
    for k in [
        "fw_before_kernel",
        "kernel_before_total",
        "total_before_issue",
        "between_issue_barrier",
        "total_after_barrier",
        "kernel_after_total",
        "fw_after_kernel",
        "other_kernel",
    ]:
        values = [r[k + "_us"] for r in every]
        text.append(f"| {k} | {min(values):.6f} | {statistics.median(values):.6f} | {max(values):.6f} |")
    text += [""]
text += [
    "## Interpretation",
    "",
    "The issue and barrier split varies substantially by core. The core with the longest barrier is not necessarily the slowest core: time spent waiting for completions can shift between command-buffer backpressure during issuance and the final barrier. A barrier-only ranking would misidentify the critical path.",
    "",
    "BRISC firmware is a surrounding control span, not an additional payload stage to add to NCRISC. There is no compute, unpack, pack or DRAM-write stage in this kernel. Tiny inter-zone gaps include marker overhead and control instructions; they cannot be interpreted as isolated hardware latency.",
    "",
    "These captures do not separate address generation, command-register programming, command-buffer stalls, DRAM service, or per-link congestion inside the issue region. No duration for those components can be extracted independently from the current scopes. Other transaction sizes and core counts are not in these two Tracy captures.",
    "",
    "[All cores and all trials CSV](all_core_durations.csv) · [Whole-program CSV](program_durations.csv). Reproduce the extraction with `python_env/bin/python tools/data_movement_perf/analyze_all_read_durations.py`.",
]
(OUT / "all_durations_report.md").write_text("\n".join(text) + "\n")
print(json.dumps(programs, indent=2))
for n in (7, 44):
    d = [r for r in all_rows if r["cores"] == n and r["selected"]]
    print(
        n,
        {
            k: [min(r[k] for r in d), statistics.median(r[k] for r in d), max(r[k] for r in d)]
            for k in ["issue_us", "barrier_us", "kernel_us", "other_kernel_us"]
        },
    )
