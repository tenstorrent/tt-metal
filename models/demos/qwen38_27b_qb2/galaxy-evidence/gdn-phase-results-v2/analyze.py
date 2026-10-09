# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce per-core phase totals from retained raw device events and signposts."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    assert values
    return dict(
        n=len(values),
        min=min(values),
        median=statistics.median(values),
        mean=statistics.mean(values),
        p90=values[min(len(values) - 1, (9 * len(values) + 9) // 10 - 1)],
        max=max(values),
    )


def analyze(root):
    ops_path = next(root.rglob("ops_perf*.csv.gz"))
    partitions_path = root / "event-partitions.json"
    partitions = json.loads(partitions_path.read_text())
    receipt = json.loads(next(root.rglob("phase.json")).read_text())
    assert receipt["passed"] and receipt["cleanup_completed"] and receipt["kernel_calls"] == 24
    calls = {}
    active = None
    with gzip.open(ops_path, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["OP TYPE"] == "signpost":
                match = re.fullmatch(r"GDN_PHASE_B(\d+)_BUFFERS(\d+)_ZONES(\d+)_STEP(\d+)_(BEGIN|END)", row["OP CODE"])
                assert match, row["OP CODE"]
                config = tuple(map(int, match.groups()[:4]))
                if match[5] == "BEGIN":
                    assert active is None
                    active = config
                else:
                    assert active == config
                    active = None
            elif active is not None:
                assert row["OP CODE"] == "GenericOpDeviceOperation"
                device, call = int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"])
                key = (device, call)
                assert key not in calls
                calls[key] = dict(
                    config=active,
                    device=device,
                    call=call,
                    kernel_us=float(row["DEVICE KERNEL DURATION [ns]"]) / 1000,
                    core_count=int(row["CORE COUNT"]),
                    counters={name: row[name] for name in ("NOC UTIL (%)", "DRAM BW UTIL (%)", "NPE CONG IMPACT (%)")},
                )
    assert active is None and len(calls) == 96
    starts, intervals, seen = {}, collections.defaultdict(list), set()
    total_rows = selected_rows = duplicates = 0
    architecture = None
    frequency_mhz = None

    def raw_rows():
        nonlocal architecture, frequency_mhz
        for name, record in sorted(partitions["files"].items()):
            path = root / "device-events" / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            with gzip.open(path, "rt") as stream:
                header = next(stream).strip()
                if architecture is not None:
                    assert architecture == header
                architecture = header
                frequency_mhz = int(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", architecture)[1])
                yield from csv.DictReader(stream, skipinitialspace=True)

    for row in raw_rows():
        total_rows += 1
        run = (int(row["PCIe slot"]), int(row["run host ID"]))
        if run not in calls:
            continue
        zone = row["zone name"]
        if not (zone.startswith("GDN_") or zone.endswith("-KERNEL")):
            continue
        assert row["type"] in ("ZONE_START", "ZONE_END")
        key = (*run, int(row["core_x"]), int(row["core_y"]), row["RISC processor type"], zone)
        clock = int(row["time[cycles since reset]"])
        event = (key, clock, row["type"])
        if event in seen:
            duplicates += 1
            continue
        seen.add(event)
        selected_rows += 1
        if row["type"] == "ZONE_START":
            assert key not in starts, ("unclosed", key)
            starts[key] = clock
        else:
            begin = starts.pop(key)
            assert clock >= begin
            intervals[key].append((begin, clock))
    assert not starts
    assert selected_rows == partitions["selected_rows"] and total_rows == selected_rows
    assert not any(value for call in calls.values() for value in call["counters"].values())
    per_core = []
    groups = collections.defaultdict(list)
    for key, spans in sorted(intervals.items()):
        device, run, x, y, risc, zone = key
        batch, buffers, profiled, step = calls[device, run]["config"]
        cycles = sum(end - begin for begin, end in spans)
        us = cycles / frequency_mhz
        per_core.append(
            dict(
                batch=batch,
                buffers=buffers,
                profiled=profiled,
                step=step,
                device=device,
                run=run,
                x=x,
                y=y,
                risc=risc,
                zone=zone,
                items=len(spans),
                total_us=us,
            )
        )
        groups[batch, buffers, profiled, risc, zone].append((us, len(spans)))
    # Every profiled work item must contain every zone on its expected RISC.
    expected_riscs = {"READER": ("NCRISC",), "WRITER": ("BRISC",), "COMPUTE": ("TRISC_0", "TRISC_1", "TRISC_2")}
    observed_riscs = sorted({row["risc"] for row in per_core})
    zones = sorted({row["zone"] for row in per_core if row["zone"].startswith("GDN_")})
    assert len(zones) == 10
    for batch in (16, 32):
        for buffers in (1, 2):
            for zone in zones:
                for risc in expected_riscs[zone.split("_")[1]]:
                    entries = groups[batch, buffers, 1, risc, zone]
                    assert sum(n for _, n in entries) == batch * 12 * 4 * 3 * 4, (
                        batch,
                        buffers,
                        risc,
                        zone,
                        sum(n for _, n in entries),
                        len(entries),
                        duplicates,
                    )
    summaries = []
    for (batch, buffers, profiled, risc, zone), values in sorted(groups.items()):
        summaries.append(
            dict(
                batch=batch,
                buffers=buffers,
                profiled=profiled,
                risc=risc,
                zone=zone,
                core_total_us=distribution([v for v, _ in values]),
                items_per_core=distribution([n for _, n in values]),
            )
        )
    kernels = []
    for config in sorted({value["config"][:3] for value in calls.values()}):
        rows = [value for value in calls.values() if value["config"][:3] == config]
        kernels.append(
            dict(
                batch=config[0],
                buffers=config[1],
                profiled=config[2],
                duration_us=distribution([row["kernel_us"] for row in rows]),
                per_rank={
                    str(d): distribution([row["kernel_us"] for row in rows if row["device"] == d])
                    for d in sorted(receipt["device_ids"])
                },
                core_counts=sorted({row["core_count"] for row in rows}),
            )
        )
    return dict(
        scope="Synthetic recurrence; three eager calls on four TP4 ranks. Includes profiler overhead.",
        limitations=[
            "RISC phases overlap; do not sum across processors.",
            "Compute regions include unpack/pack synchronization; they are not pure arithmetic time.",
            "DRAM regions include issue and completion waits; they are not measured link utilization.",
            "No NoC congestion counters were populated.",
            "Two input buffers are already used by the model candidate; that gain is not new.",
        ],
        architecture=architecture,
        frequency_mhz=frequency_mhz,
        total_raw_rows=partitions["selected_rows"] + partitions["excluded_rows"],
        selected_raw_rows=selected_rows,
        duplicate_raw_events=duplicates,
        observed_riscs=observed_riscs,
        all_phase_pairs_complete=True,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ops_path, partitions_path)
        },
        kernels=kernels,
        phase_summaries=summaries,
        per_core=per_core,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for row in report["kernels"]:
        print(row["batch"], row["buffers"], row["profiled"], row["duration_us"])
