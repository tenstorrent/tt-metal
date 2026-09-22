# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Count recorded NoC payloads in focused MLP windows, with coverage gates.

This reports instrumented API requests, not DRAM-controller bus counters or
bandwidth. Payloads have32-byte resolution. Stateful requests are resolved
from preceding set-state events; missing state/capped payloads invalidate totals.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import yaml


def count_events(events, dram_coordinates, payload_cap):
    states = {}
    zones = Counter()
    totals = Counter()
    types = Counter()
    for event in sorted(events, key=lambda e: e["timestamp"]):
        core = (event.get("src_device_id"), event.get("sx"), event.get("sy"), event.get("proc"))
        if "zone" in event:
            if event["zone"].endswith("-KERNEL"):
                key = (*core, event["zone"])
                if event["zone_phase"] == "ZONE_START":
                    zones[key] += 1
                    totals["kernel_starts"] += 1
                elif event["zone_phase"] == "ZONE_END":
                    zones[key] -= 1
                    totals["kernel_ends"] += 1
                    if zones[key] < 0:
                        totals["orphan_kernel_ends"] += 1
            continue
        kind = event.get("type", "")
        types[kind] += 1
        direction = "read" if kind.startswith("READ") else "write" if kind.startswith("WRITE") else None
        if direction is None:
            continue
        key = (*core, event.get("noc"), direction)
        size = int(event.get("num_bytes", 0))
        endpoint = (event["dx"], event["dy"]) if "dx" in event and "dy" in event else None
        if kind.endswith("SET_STATE"):
            states[key] = (endpoint, size)
            continue
        if kind not in {
            "READ",
            "READ_WITH_STATE",
            "READ_WITH_STATE_AND_TRID",
            "READ_DRAM_SHARDED_WITH_STATE",
            "WRITE_",
            "WRITE_WITH_TRID",
            "WRITE_WITH_STATE",
            "WRITE_WITH_TRID_WITH_STATE",
        }:
            if kind == "WRITE_MULTICAST":
                totals["multicast_payload_bytes"] += size
            continue
        if "WITH_STATE" in kind:
            saved_endpoint, saved_size = states.get(key, (None, 0))
            endpoint = endpoint if endpoint is not None else saved_endpoint
            size = size or saved_size
        else:
            # Direct requests overwrite the hardware command buffer. Its final
            # packet state is not in this format; require a new explicit state
            # before attributing subsequent stateful transfers.
            states.pop(key, None)
        totals["requests"] += 1
        if endpoint is None or size == 0:
            totals["unresolved_requests"] += 1
            continue
        if size >= payload_cap:
            totals["possibly_saturated_requests"] += 1
        site = "dram" if endpoint in dram_coordinates else "other_endpoint"
        totals[f"{site}_{direction}_payload_bytes"] += size
        totals[f"{site}_{direction}_requests"] += 1
    totals["unbalanced_kernel_zones"] = sum(abs(v) for v in zones.values())
    totals["missing_kernel_zones"] = int(not totals["kernel_starts"])
    return dict(totals), dict(types)


def analyze(logs, ops_csv, *, extended_payload=False, expected_dram_read_bytes=None):
    soc_path = logs / "soc_descriptor.yaml"
    soc = yaml.safe_load(soc_path.read_text())
    dram = {tuple(map(int, coordinate.split("-"))) for channel in soc["dram"] for coordinate in channel}
    cap = 1048544 if extended_payload else 8160
    windows, window = [], None
    with ops_csv.open() as stream:
        for row in csv.DictReader(stream):
            name = row["OP CODE"].strip()
            if name.startswith("QB2_TRAFFIC_BEGIN_"):
                if window is not None:
                    raise ValueError("Nested traffic windows")
                window = {"repeat": int(name.rsplit("_", 1)[1]), "operations": [], "totals_per_device": {}}
            elif name.startswith("QB2_TRAFFIC_END_"):
                if window is None or window["repeat"] != int(name.rsplit("_", 1)[1]):
                    raise ValueError("Unmatched traffic signpost")
                windows.append(window)
                window = None
            elif window is not None and row.get("DEVICE ID", "").strip():
                device = int(row["DEVICE ID"])
                runtime = int(row["GLOBAL CALL COUNT"]) & 0xFFFFFFFF
                trace = int(row["METAL TRACE ID"])
                replay = int(row["METAL TRACE REPLAY SESSION ID"])
                candidates = list(
                    logs.glob(f"noc_trace_dev{device}_*_ID{runtime}_traceID{(trace << 32) | replay}.json")
                )
                entry = {"device": device, "runtime_id": runtime, "trace": trace, "replay": replay, "op": name}
                if len(candidates) != 1:
                    entry["missing_or_ambiguous_capture"] = [str(p) for p in candidates]
                else:
                    entry["file"] = str(candidates[0])
                    counts, types = count_events(json.loads(candidates[0].read_text()), dram, cap)
                    counts["missing_device_duration"] = int(not row.get("DEVICE KERNEL DURATION [ns]", "").strip())
                    counts["unsupported_events"] = types.get("UNSUPPORTED", 0)
                    entry.update({"counts": counts, "event_types": types})
                    aggregate = Counter(window["totals_per_device"].get(str(device), {}))
                    aggregate.update(counts)
                    window["totals_per_device"][str(device)] = dict(aggregate)
                window["operations"].append(entry)
    if window is not None or not windows:
        raise ValueError("Missing or unfinished traffic windows")
    disqualifiers = (
        "unresolved_requests",
        "possibly_saturated_requests",
        "unbalanced_kernel_zones",
        "orphan_kernel_ends",
        "missing_kernel_zones",
        "missing_device_duration",
        "unsupported_events",
    )
    for window in windows:
        window["operations_per_device"] = dict(Counter(o["device"] for o in window["operations"]))
        window["expected_dram_read_bytes_per_device"] = expected_dram_read_bytes
        window["expected_payload_matches"] = expected_dram_read_bytes is None or all(
            counts.get("dram_read_payload_bytes", 0) == expected_dram_read_bytes
            for counts in window["totals_per_device"].values()
        )
        window["complete"] = (
            window["expected_payload_matches"]
            and set(window["operations_per_device"]) == {0, 1, 2, 3}
            and len(set(window["operations_per_device"].values())) == 1
            and all(
                "counts" in o and not any(o["counts"].get(k, 0) for k in disqualifiers) for o in window["operations"]
            )
        )
    return {
        "measurement": "Recorded NoC API payload bytes,32B resolution; not DRAM bus bytes or latency",
        "requires_log_review": "Reject profiler dropped-event warnings even when kernel endpoints are balanced",
        "payload_encoding_cap": cap,
        "soc_descriptor": str(soc_path),
        "complete": all(w["complete"] for w in windows),
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--ops-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extended-payload", action="store_true", help="Only for the matching extended payload runtime"
    )
    parser.add_argument("--expected-dram-read-bytes", type=int, help="Independent expected bytes per chip/window")
    args = parser.parse_args()
    result = analyze(
        args.logs,
        args.ops_csv,
        extended_payload=args.extended_payload,
        expected_dram_read_bytes=args.expected_dram_read_bytes,
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=2))
    if not result["complete"]:
        raise SystemExit("Traffic capture failed coverage or payload-resolution gates; inspect the saved report")


if __name__ == "__main__":
    main()
