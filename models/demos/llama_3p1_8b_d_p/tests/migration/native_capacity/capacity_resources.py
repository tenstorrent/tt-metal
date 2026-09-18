"""Small OS-only admission check; charged cgroup bytes are not process RSS."""

import shutil
from pathlib import Path

from capacity_execution import require


def admit(snapshot, minimum_memory, minimum_disk):
    require(snapshot["mem_available_bytes"] >= minimum_memory, "Host MemAvailable below reviewed admission floor")
    require(snapshot["disk_free_bytes"] >= minimum_disk, "Shared disk below reviewed admission floor")
    finite = []
    for group in snapshot["cgroups"]:
        for kind in ("max", "high"):
            value = group[kind]
            if value is not None:
                remaining = max(0, value - group["current"])
                finite.append(remaining)
                require(
                    remaining >= minimum_memory,
                    "Finite cgroup charged headroom below floor; current includes pagecache, review reclaimability",
                )
    return dict(
        host_memory_floor_bytes=minimum_memory,
        disk_floor_bytes=minimum_disk,
        conservative_cgroup_charged_headroom_bytes=min(finite) if finite else None,
        cgroup_current_is_process_rss=False,
    )


def observe(shared, *, proc=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    values = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, _, raw = line.partition(":")
        if key in ("MemAvailable", "MemTotal"):
            fields = raw.split()
            require(len(fields) == 2 and fields[1] == "kB", "Malformed host memory unit")
            values[key] = int(fields[0]) * 1024
    require(set(values) == {"MemAvailable", "MemTotal"}, "Missing host memory")
    memberships = [x.split(":", 2)[2] for x in (proc / "self/cgroup").read_text().splitlines() if x.startswith("0::")]
    require(len(memberships) == 1, "Unified cgroup membership unavailable")
    current = cgroup_root / memberships[0].lstrip("/")
    require(current.resolve().is_relative_to(cgroup_root.resolve()), "Cgroup path escapes hierarchy")
    groups = []
    while True:
        if (current / "memory.current").exists():
            limits = {}
            for field in ("max", "high"):
                raw = (
                    (current / ("memory." + field)).read_text().strip()
                    if (current / ("memory." + field)).exists()
                    else "max"
                )
                limits[field] = None if raw == "max" else int(raw)
            stat = dict(line.split() for line in (current / "memory.stat").read_text().splitlines())
            groups.append(
                dict(
                    path=str(current),
                    current=int((current / "memory.current").read_text()),
                    file_bytes=int(stat.get("file", 0)),
                    inactive_file_bytes=int(stat.get("inactive_file", 0)),
                    **limits,
                )
            )
        if current == cgroup_root:
            break
        current = current.parent
    require(groups, "No cgroup memory observations")
    return dict(
        mem_available_bytes=values["MemAvailable"],
        mem_total_bytes=values["MemTotal"],
        cgroups=groups,
        disk_free_bytes=shutil.disk_usage(shared).free,
        current_scope="cgroup charges including file cache; not unreclaimable RSS",
    )
