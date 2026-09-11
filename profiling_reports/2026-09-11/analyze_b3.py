"""Recompute accumulated device work from committed reduced captures; no hardware."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / "models/demos/deepseek_v3_d_p/tests/perf/captures"
SOCKET = {"InboundSocketServiceSyncOperation", "OutboundSocketServiceSyncOperation"}
results = {}
for path in sorted(CAPTURES.glob("ops_*.csv.gz")):
    with gzip.open(path, "rt") as stream:
        rows = [row for row in csv.DictReader(stream) if row["DEVICE ID"]]
    devices = {row["DEVICE ID"] for row in rows}
    passes = collections.Counter(row["DEVICE ID"] for row in rows if row["OP CODE"] == "RingJointSDPADeviceOperation")
    assert set(passes) == devices and set(passes.values()) == {9}
    sums = collections.defaultdict(float)
    for row in rows:
        duration = float(row["DEVICE KERNEL DURATION [ns]"])
        assert duration >= 0
        if row["OP CODE"] not in SOCKET:
            sums[row["OP CODE"]] += duration / (1e6 * len(devices) * 9)
    total = sum(sums.values())
    results[path.name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "devices": len(devices),
        "sdpa_invocations_per_device": 9,
        "metric": "sum of all non-socket device kernel durations / devices / 9; accumulated work, not elapsed time",
        "total_ms": total,
        "operations": {
            op: {"ms": ms, "percent": 100 * ms / total} for op, ms in sorted(sums.items(), key=lambda item: -item[1])
        },
    }
print(json.dumps(results, indent=2))
