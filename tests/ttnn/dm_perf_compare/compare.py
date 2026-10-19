# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Turns the per-round raw measurements into the results document. Pure; no device, no ttnn."""

import stats
from ops import CONTROL

SCHEMA = "dm-perf-compare:v1"


def series(runs, call_id, impl):
    """One implementation's per-round best times plus its status and output digests across rounds."""
    values, digests, status, error = [], set(), "ok", None
    for run in runs:
        entry = None if run is None else run["calls"].get(call_id, {}).get("results", {}).get(impl)
        if entry is None:
            values.append(None)
            if status == "ok":
                status = "missing"
        elif entry["status"] == "ok":
            values.append(entry["best_us"])
            digests.add(entry["digest"])
        else:
            values.append(None)
            if status in ("ok", "missing"):
                status, error = entry["status"], entry.get("error")
    return {"us": values, "status": status, "error": error, "digests": sorted(digests)}


def _from_changes(changes, noise):
    if not changes:
        return {"comparable": False, "reason": "no round measured on both sides"}
    return {
        "comparable": True,
        "changes_pct": changes,
        "median_pct": stats.median(changes),
        "verdict": stats.verdict(changes, noise),
    }


def _comparison(base, other, noise):
    """`other` against `base`; the verdict is about `other` (faster means other beats base)."""
    if base["status"] != "ok" or other["status"] != "ok":
        reason = base if base["status"] != "ok" else other
        return {"comparable": False, "reason": "%s: %s" % (reason["status"], reason["error"] or "no data")}
    if len(base["digests"]) != 1 or base["digests"] != other["digests"]:
        return {"comparable": False, "reason": "output differs"}
    return _from_changes(stats.per_round_changes(base["us"], other["us"]), noise)


def _host_detail(base, other, resolution):
    """Host-time change, reported as moved/not against the run's host resolution, never as a verdict."""
    found = _comparison(base, other, resolution)
    if found["comparable"]:
        found["moved"] = found.pop("verdict") != "no change"
    return found


def _device_comparison(host, base_ns, other_ns, noise):
    """Device time of `other` against `base`; host checks (status, output match) gate it."""
    if not host["comparable"] and not host["reason"].startswith("no round"):
        return host
    if not any(v is not None for v in base_ns) or not any(v is not None for v in other_ns):
        return {"comparable": False, "reason": "no device time"}
    return _from_changes(stats.per_round_changes(base_ns, other_ns), noise)


def _device(device_rounds, index, impl):
    samples = [r[(impl, index)] for r in device_rounds if (impl, index) in r]
    if not samples:
        return None
    return {
        "ns_rounds": [s["ns"] for s in samples],
        "ns": stats.median([s["ns"] for s in samples]),
        "ops": samples[0]["ops"],
    }


def _device_rounds(device_rounds, index, impl):
    """Per-round device ns, None where that round has no sample, so rounds stay paired."""
    return [r[(impl, index)]["ns"] if (impl, index) in r else None for r in device_rounds]


def _default_uses(device):
    """Which implementation the default ran, read off the device ops it dispatched."""
    default = device.get("default")
    if default is None:
        return None
    ops = sorted(default["ops"])
    matches = [i for i in ("native", "codegen") if device.get(i) and sorted(device[i]["ops"]) == ops]
    if len(matches) == 1:
        return matches[0]
    return "indistinguishable" if matches else "other: " + ", ".join(sorted(set(default["ops"])))


def measure_noise(raw, calls):
    """Host resolution from the same-tree control over every call; device band from the after
    build's own device rounds against each other (adjacent rounds, free: no extra process)."""
    host, device = {}, {}
    for index, call in enumerate(calls):
        base = series(raw["time"]["ctl_before"], call["id"], "default")
        other = series(raw["time"]["ctl_after"], call["id"], "default")
        if base["status"] == "ok" and other["status"] == "ok":
            host[call["id"]] = stats.per_round_changes(base["us"], other["us"])
        ns = _device_rounds(raw["device"]["after"], index, "default")
        # Absolute values: random noise gives adjacent changes of both signs, whose median cancels to zero.
        changes = [abs(c) for c in stats.per_round_changes(ns[:-1], ns[1:])]
        if changes:
            device[call["id"]] = changes
    device_band = stats.noise_pct(device, stats.DEVICE_MIN_PCT)
    return {
        "host_resolution_pct": stats.noise_pct(host),
        "host_per_call_changes_pct": host,
        "device_band_pct": device_band,
        "device_per_call_changes_pct": device,
        "reliable": device_band <= stats.MAX_NOISE_PCT,
    }


def build(calls_doc, raw, meta):
    """`raw["time"]`: runs per label (`before`, `after_default`, `after`, `ctl_before`, `ctl_after`:
    one worker JSON or None per round); `before` and `after_default` run the default only and give
    the host-time detail, `after` also runs the other implementations and the control copy of the
    default. `raw["device"]`: `before` (default only) and `after` (every implementation), each per
    round `{(impl, call index): {"ns", "ops"}}`. The Q1 and Q2 verdicts read device time.
    """
    noise = measure_noise(raw, calls_doc["calls"])
    host_pct, device_pct = noise["host_resolution_pct"], noise["device_band_pct"]
    rows, skipped = [], []
    for index, call in enumerate(calls_doc["calls"]):
        cid = call["id"]
        before = {"default": series(raw["time"]["before"], cid, "default")}
        after = {impl: series(raw["time"]["after"], cid, impl) for impl in ("default", "native", "codegen", CONTROL)}
        after_solo = series(raw["time"]["after_default"], cid, "default")
        q1_host = _host_detail(before["default"], after_solo, host_pct)
        q1 = _device_comparison(
            q1_host,
            _device_rounds(raw["device"]["before"], index, "default"),
            _device_rounds(raw["device"]["after"], index, "default"),
            device_pct,
        )
        q2_host = {impl: _comparison(after["default"], after[impl], host_pct) for impl in ("native", "codegen")}
        # The default's public entry and routing check cost a few microseconds the forced entries
        # skip, so on host time it never wins; device time is where the choice of implementation shows.
        default_ns = _device_rounds(raw["device"]["after"], index, "default")
        q2 = {
            impl: _device_comparison(
                q2_host[impl], default_ns, _device_rounds(raw["device"]["after"], index, impl), device_pct
            )
            for impl in ("native", "codegen")
        }
        q2_control = _comparison(after["default"], after[CONTROL], host_pct)
        beaten_by = [i for i, c in q2.items() if c["comparable"] and c["verdict"] == "faster"]
        device = {}
        for impl in ("default", "native", "codegen"):
            found = _device(raw["device"]["after"], index, impl)
            if found:
                device[impl] = found
        device_before = _device(raw["device"]["before"], index, "default")
        for side, impls in (("before", before), ("after", after)):
            for impl, s in impls.items():
                if s["status"] in ("refused", "error"):
                    skipped.append(
                        {"call": cid, "side": side, "impl": impl, "status": s["status"], "error": s["error"]}
                    )
        rows.append(
            {
                "id": cid,
                "call": {k: call[k] for k in ("shape", "dtype", "layout", "memory_config", "args")},
                "before": before,
                "after": after,
                "after_default_only": after_solo,
                "q1": q1,
                "q1_host": q1_host,
                "q2": q2,
                "q2_host": q2_host,
                "q2_control": q2_control,
                "fastest_option_used": None if not any(c["comparable"] for c in q2.values()) else not beaten_by,
                "beaten_by": beaten_by,
                "device": device,
                "device_before": device_before,
                "default_uses": _default_uses(device) if device else None,
            }
        )
    return {"schema": SCHEMA, **meta, "noise": noise, "calls": rows, "skipped": skipped, "summary": _summary(rows)}


def _summary(rows):
    q1 = {"faster": 0, "slower": 0, "no change": 0, "not comparable": 0}
    changes = []
    for row in rows:
        if row["q1"]["comparable"]:
            q1[row["q1"]["verdict"]] += 1
            changes.append(row["q1"]["median_pct"])
        else:
            q1["not comparable"] += 1
    host_moved = sum(1 for row in rows if row["q1_host"]["comparable"] and row["q1_host"]["moved"])
    q2 = {"default fastest": 0, "beaten": 0, "not comparable": 0}
    for row in rows:
        used = row["fastest_option_used"]
        q2["not comparable" if used is None else "default fastest" if used else "beaten"] += 1
    # Two identical callables through the Q2 rule: anything but "no change" is a false flag.
    q2_control = {"faster": 0, "slower": 0, "no change": 0, "not comparable": 0}
    for row in rows:
        control = row["q2_control"]
        q2_control[control["verdict"] if control["comparable"] else "not comparable"] += 1
    return {
        "q1": q1,
        "q1_geomean_pct": stats.geomean_change_pct(changes),
        "q1_host_moved": host_moved,
        "q2": q2,
        "q2_control": q2_control,
    }
