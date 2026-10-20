import re
import statistics
import sys
from datetime import datetime
from pathlib import Path


log = Path(sys.argv[1]).read_text()
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
timestamp = r"\[(?P<ts>[^]]+)\]"
starts = {}
ends = {}
positions = {}
for line in log.splitlines():
    start = re.search(
        timestamp
        + r".*migrate command: .*migration_id=(?P<id>\d+).*src_positions=\[(?P<begin>\d+),(?P<end>\d+)\)",
        line,
    )
    if start:
        migration_id = int(start["id"])
        starts.setdefault(migration_id, datetime.fromisoformat(start["ts"]))
        positions[migration_id] = (int(start["begin"]), int(start["end"]))
    end = re.search(
        timestamp + r".*migration_id=(?P<id>\d+).*completed successful=true reason= releasing", line
    )
    if end:
        ends[int(end["id"])] = datetime.fromisoformat(end["ts"])

durations_ms = {
    migration_id: (ends[migration_id] - start).total_seconds() * 1000
    for migration_id, start in starts.items()
    if migration_id in ends
}
if limit:
    keep = set(sorted(durations_ms)[-limit:])
    starts = {key: value for key, value in starts.items() if key in keep}
    ends = {key: value for key, value in ends.items() if key in keep}
    positions = {key: value for key, value in positions.items() if key in keep}
    durations_ms = {key: value for key, value in durations_ms.items() if key in keep}

if not durations_ms:
    raise SystemExit("no successful migrations found")

print(f"transfers={len(durations_ms)}")
print(f"sum_ms={sum(durations_ms.values()):.1f}")
print(f"median_ms={statistics.median(durations_ms.values()):.1f}")
print(f"span_ms={(max(ends.values()) - min(starts.values())).total_seconds() * 1000:.1f}")
for position in sorted(set(positions.values())):
    ids = [migration_id for migration_id, value in positions.items() if value == position and migration_id in ends]
    chunk_span = (max(ends[i] for i in ids) - min(starts[i] for i in ids)).total_seconds() * 1000
    print(f"chunk={position[0]}:{position[1]} transfers={len(ids)} span_ms={chunk_span:.1f}")
