"""Capacity table and memory admission checks. No TTNN imports or device access."""

import re
from pathlib import Path

from capacity_contract import Geometry
from native_observation import process_identity


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_capacity_ready(capacity, role, host, health, log):
    """Require all table loads and local indexes before accepting native readiness."""
    geometry = Geometry(capacity)
    require(capacity <= 65536, "128K testing is deferred")
    require(role in ("source", "passive") and isinstance(host, str) and host, "Invalid endpoint")
    require(health == 200, "Native health is not ready")
    require(
        log.count("Device map loaded:") == 1 and log.count("Device map loaded: 32 entries") == 1,
        "Incomplete or duplicate device map",
    )
    label = "prefill" if role == "source" else "decode"
    entries = geometry.table_entries
    index = (
        f"KV chunk indexes installed for host '{host}' from {label} tables "
        f"(16 configs, {entries} read / {entries} write chunks over 32 device(s))"
    )
    require(
        log.count("KV chunk indexes installed for host ") == 1 and log.count(index) == 1,
        "Local index geometry or endpoint differs",
    )
    loads = []
    for prefix in ("Prefill", "Decode"):
        marker = f"{prefix} KV chunk table loaded: config '"
        require(log.count(marker) == 16, "Missing or extra native table configuration")
        for kind in ("k", "v"):
            for head in range(8):
                text = marker + f"{kind}_h{head}'"
                require(log.count(text) == 1, "Missing or duplicate native table configuration")
                loads.append(log.index(text))
    final = "All configured tables loaded successfully"
    require(log.count(final) == 1, "Missing or duplicate table completion")
    require(max(loads) < log.index(index) < log.index(final), "Table readiness order differs")
    allowed = "KV manager not ready after startup (discovery pending: peers not yet resolved)"
    for line in log.splitlines():
        if "[ERROR]" in line:
            require(allowed in line, "Unexpected native startup error")
    return dict(
        capacity=capacity,
        role=role,
        host=host,
        configs=16,
        devices=32,
        read_entries=entries,
        write_entries=entries,
        health=200,
    )


def observe_manager_memory(process, expected_identity, limit_bytes, *, proc=Path("/proc")):
    """Bracket RSS and lifetime peak memory with the launched manager's generation."""
    require(type(limit_bytes) is int and limit_bytes > 0, "A positive memory budget is required")
    require(process.poll() is None, "Manager already exited")
    before = process_identity(process.pid, proc)
    require(
        all(
            type(expected_identity.get(key)) is int
            and expected_identity[key] > 0
            and before.get(key) == expected_identity[key]
            for key in ("pid", "start_ticks")
        ),
        "Manager differs from the launched generation",
    )
    require(before.get("state") not in ("Z", "X"), "Manager is not live")
    status = (Path(proc) / str(process.pid) / "status").read_text()
    values = {}
    for field in ("VmRSS", "VmHWM"):
        rows = [line for line in status.splitlines() if line.startswith(field + ":")]
        require(len(rows) == 1, "Missing or duplicate memory field")
        match = re.fullmatch(field + r":\s+([0-9]+)\s+kB\s*", rows[0])
        require(match is not None, "Malformed memory field or unit")
        values[field] = int(match.group(1)) * 1024
        require(0 < values[field] <= limit_bytes, "Manager memory exceeds budget or is zero")
    require(values["VmHWM"] >= values["VmRSS"], "Peak memory is smaller than current memory")
    after = process_identity(process.pid, proc)
    require(
        all(before.get(key) == after.get(key) for key in ("pid", "start_ticks"))
        and after.get("state") not in ("Z", "X")
        and process.poll() is None,
        "Manager changed or exited during memory observation",
    )
    return dict(
        process=after,
        rss_bytes=values["VmRSS"],
        hwm_bytes=values["VmHWM"],
        limit_bytes=limit_bytes,
        scope="Whole manager lifetime peak, not per-index allocation",
    )
