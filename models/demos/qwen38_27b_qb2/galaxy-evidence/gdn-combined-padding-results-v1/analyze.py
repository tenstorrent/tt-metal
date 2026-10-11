# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair current GDN phase intervals with exact device/call IDs and signposts."""
import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError("Empty timing distribution")
    return dict(
        n=len(values),
        min=values[0],
        median=statistics.median(values),
        mean=statistics.mean(values),
        p90=values[(9 * len(values) + 9) // 10 - 1],
        max=values[-1],
    )


def analyze(root):
    ops_path = next((root / "tracy/reports").glob("*/ops_perf_results*.csv"))
    raw_path = root / "tracy/.logs/profile_log_device.csv"
    receipt = json.loads((root / "phase.json").read_text())
    if not receipt["passed"] or not receipt["cleanup_completed"] or receipt["kernel_calls"] != 48:
        raise ValueError("Incomplete or failing phase receipt")
    calls = {}
    active = None
    counts = None
    with ops_path.open() as stream:
        for row in csv.DictReader(stream):
            if row["OP TYPE"] == "signpost":
                match = re.fullmatch(
                    r"GDN_PIPELINE_B(\d+)_(ZERO|SKIP)_ZONES(\d+)_STEP(\d+)_(BEGIN|END)", row["OP CODE"]
                )
                if not match:
                    raise ValueError("Unexpected signpost " + row["OP CODE"])
                config = (int(match[1]), match[2].lower(), int(match[3]), int(match[4]))
                if match[5] == "BEGIN":
                    if active is not None:
                        raise ValueError("Nested signpost")
                    active = config
                    counts = collections.Counter()
                else:
                    if active != config or set(counts) != set(receipt["device_ids"]) or set(counts.values()) != {2}:
                        raise ValueError("Incomplete two-kernel/four-rank pipeline")
                    active = None
            elif active is not None:
                if row["OP CODE"] != "GenericOpDeviceOperation":
                    raise ValueError("Unexpected pipeline operation")
                device, call = int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"])
                key = (device, call)
                if key in calls or counts[device] > 1:
                    raise ValueError("Duplicate pipeline operation")
                kind = ("recurrence", "epilogue")[counts[device]]
                counts[device] += 1
                calls[key] = dict(
                    config=active,
                    kind=kind,
                    kernel_us=float(row["DEVICE KERNEL DURATION [ns]"]) / 1000,
                    cores=int(row["CORE COUNT"]),
                )
    if active is not None or len(calls) != 192:
        raise ValueError("Incomplete pipeline calls")
    starts = {}
    last = {}
    totals = collections.defaultdict(lambda: [0, 0])
    selected = excluded = 0
    with raw_path.open() as stream:
        architecture = next(stream).strip()
        frequency = int(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", architecture)[1])
        for row in csv.DictReader(stream, skipinitialspace=True):
            run = (int(row["PCIe slot"]), int(row["run host ID"]))
            zone = row["zone name"]
            if run not in calls or not (zone.startswith("GDN_") or zone.endswith("-KERNEL")):
                excluded += 1
                continue
            event = row["type"]
            clock = int(row["time[cycles since reset]"])
            key = (*run, int(row["core_x"]), int(row["core_y"]), row["RISC processor type"], zone)
            if event not in ("ZONE_START", "ZONE_END") or clock <= last.get(key, -1):
                raise ValueError("Duplicate/out-of-order phase event " + str(key))
            last[key] = clock
            selected += 1
            if event == "ZONE_START":
                if key in starts:
                    raise ValueError("Unclosed phase " + str(key))
                starts[key] = clock
            else:
                if key not in starts:
                    raise ValueError("Unmatched phase end " + str(key))
                begin = starts.pop(key)
                totals[key][0] += clock - begin
                totals[key][1] += 1
    if starts:
        raise ValueError("Incomplete raw phase pairs")
    groups = collections.defaultdict(list)
    for (device, call, x, y, risc, zone), (cycles, items) in totals.items():
        c = calls[device, call]
        batch, padding, profiled, step = c["config"]
        groups[batch, padding, profiled, c["kind"], risc, zone].append((cycles / frequency, items))
    zones = sorted({key[-1] for key in groups if key[-1].startswith("GDN_")})
    if len(zones) != 24:
        raise ValueError("Missing current kernel zones")
    for batch in (16, 32):
        for padding in ("zero", "skip"):
            for zone in zones:
                kind = "epilogue" if zone.startswith("GDN_EP_") else "recurrence"
                riscs = (
                    ("NCRISC",)
                    if "READER" in zone
                    else ("BRISC",)
                    if "WRITER" in zone
                    else ("TRISC_0", "TRISC_1", "TRISC_2")
                )
                for risc in riscs:
                    observed = sum(items for _, items in groups[batch, padding, 1, kind, risc, zone])
                    expected = batch * 12 * 4 * 3 * (4 if kind == "recurrence" else 1)
                    if zone in ("GDN_EP_READER_WEIGHT", "GDN_EP_READER_PADDING"):
                        expected = sum(
                            c["cores"]
                            for c in calls.values()
                            if c["config"][:3] == (batch, padding, 1) and c["kind"] == kind
                        )
                    if observed != expected:
                        raise ValueError(f"Phase count {batch}/{padding}/{risc}/{zone}: {observed} != {expected}")
    phases = []
    for (batch, padding, profiled, kind, risc, zone), values in sorted(groups.items()):
        phases.append(
            dict(
                batch=batch,
                padding=padding,
                profiled=profiled,
                kind=kind,
                risc=risc,
                zone=zone,
                core_total_us=distribution([v[0] for v in values]),
                items_per_core=distribution([v[1] for v in values]),
            )
        )
    kernels = []
    for batch, padding, profiled, kind in sorted({(*c["config"][:3], c["kind"]) for c in calls.values()}):
        rows = [c for c in calls.values() if c["config"][:3] == (batch, padding, profiled) and c["kind"] == kind]
        kernels.append(
            dict(
                batch=batch,
                padding=padding,
                profiled=profiled,
                kind=kind,
                duration_us=distribution([c["kernel_us"] for c in rows]),
            )
        )

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                h.update(block)
        return h.hexdigest()

    return dict(
        scope="Synthetic current GDN pipeline; three eager calls, four ranks; not serving throughput",
        limitations=[
            "RISC phases overlap; never add across processors",
            "Math zones include synchronization; no hardware activity counters",
            "DMA intervals include issue and wait; no physical DRAM utilization claim",
            "Eager diagnostic placement is not a full-model critical path",
        ],
        architecture=architecture,
        frequency_mhz=frequency,
        selected_raw_rows=selected,
        excluded_raw_rows=excluded,
        all_phase_pairs_complete=True,
        all_expected_item_counts_match=True,
        source_sha256={str(p): digest(p) for p in (ops_path, raw_path, root / "phase.json")},
        kernels=kernels,
        phase_summaries=phases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(selected_raw_rows=result["selected_raw_rows"], kernels=result["kernels"]), indent=2))
