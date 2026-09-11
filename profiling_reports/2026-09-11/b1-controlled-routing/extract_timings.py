"""Validate signposted eager iterations and retain unmerged per-device timings."""
import argparse
import csv
import gzip
import io
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

OPS = ("DispatchDeviceOperation", "CombineDeviceOperation")
MARK = re.compile(r"B1_(START|END) layer=(\d+) mode=(\w+) iteration=(\d+)$")


def extract(path, expected):
    windows = []
    current = None
    identity = None
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    with io.StringIO(raw.decode(), newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"OP CODE", "OP TYPE", "DEVICE ID", "GLOBAL CALL COUNT", "DEVICE KERNEL DURATION [ns]"}
        assert required <= set(reader.fieldnames), required - set(reader.fieldnames)
        for line, row in enumerate(reader, 2):
            op = row["OP CODE"]
            match = MARK.fullmatch(op)
            if match:
                assert row["OP TYPE"] == "signpost", (line, row)
                kind, layer, mode, iteration = match.groups()
                ident = (int(layer), mode)
                assert identity is None or identity == ident, "mixed workload identity"
                identity = ident
                iteration = int(iteration)
                if kind == "START":
                    assert current is None and iteration == len(windows), (line, "nonsequential start")
                    current = {"iteration": iteration, "rows": {}, "sequence": {}}
                else:
                    assert current is not None and iteration == current["iteration"], (line, "unmatched end")
                    assert set(current["sequence"]) == set(range(8)), "missing devices"
                    for device in range(8):
                        assert current["sequence"][device] == list(OPS), (iteration, device, "unexpected sequence")
                    windows.append(current)
                    current = None
                continue
            assert not op.startswith(("B1_START", "B1_END")), (line, "malformed B1 signpost")
            if op not in OPS:
                continue
            assert current is not None, (line, "target op outside B1 window")
            device, call = int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"])
            duration = float(row["DEVICE KERNEL DURATION [ns]"])
            assert math.isfinite(duration) and duration > 0, (line, duration)
            key = (device, op)
            assert key not in current["rows"], (line, "duplicate operation")
            current["rows"][key] = (call, duration)
            current["sequence"].setdefault(device, []).append(op)
    assert current is None and len(windows) == expected, ("missing boundary/iteration", len(windows))
    # This report's GLOBAL CALL COUNT encodes device ID in low bits. Validate
    # observed call-device agreement rather than require raw IDs to be equal.
    previous_base = -1
    for window in windows:
        for op in OPS:
            bases = {window["rows"][(d, op)][0] - d for d in range(8)}
            assert len(bases) == 1, (window["iteration"], op, "inconsistent device call mapping")
            base = bases.pop()
            assert base > previous_base, "nonmonotonic operation calls"
            previous_base = base
    result = {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "layer": identity[0],
        "mode": identity[1],
        "discarded_iteration": 0,
        "validated_iterations": expected,
        "validated_devices": list(range(8)),
        "validated_target_rows": expected * 8 * 2,
        "mapping": "Pair B1_START/B1_END signposts; require one Dispatch then Combine per device/window; verify call-device same across devices for each operation and increasing across windows.",
        "interpretation": "Eager isolated kernel durations, not model latency; excludes FFN/shared expert/layout conversion/host overhead.",
        "operations": {},
    }
    for op in OPS:
        per_device = {d: [w["rows"][(d, op)][1] / 1e6 for w in windows[1:]] for d in range(8)}
        maximums = [max(per_device[d][i] for d in range(8)) for i in range(expected - 1)]
        result["operations"][op] = {
            "per_device_retained_ms": per_device,
            "per_device_mean_ms": {d: statistics.mean(v) for d, v in per_device.items()},
            "per_iteration_device_max_ms": maximums,
            "mean_of_device_max_ms": statistics.mean(maximums),
            "median_of_device_max_ms": statistics.median(maximums),
            "p90_nearest_rank_device_max_ms": sorted(maximums)[math.ceil(0.9 * len(maximums)) - 1],
            "raw_call_ids_by_iteration": [[w["rows"][(d, op)][0] for d in range(8)] for w in windows],
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert args.iterations >= 2
    args.output.write_text(json.dumps(extract(args.csv, args.iterations), indent=2) + "\n")
