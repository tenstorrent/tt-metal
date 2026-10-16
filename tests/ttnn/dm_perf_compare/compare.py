# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Turns the per-round raw measurements into the results document. Pure; no device, no ttnn."""

import stats

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


def _comparison(base, other, noise):
    """`other` against `base`; the verdict is about `other` (faster means other beats base)."""
    if base["status"] != "ok" or other["status"] != "ok":
        reason = base if base["status"] != "ok" else other
        return {"comparable": False, "reason": "%s: %s" % (reason["status"], reason["error"] or "no data")}
    if len(base["digests"]) != 1 or base["digests"] != other["digests"]:
        return {"comparable": False, "reason": "output differs"}
    changes = stats.per_round_changes(base["us"], other["us"])
    if not changes:
        return {"comparable": False, "reason": "no round measured on both sides"}
    return {
        "comparable": True,
        "changes_pct": changes,
        "median_pct": stats.median(changes),
        "verdict": stats.verdict(changes, noise),
    }


def _device(device_rounds, index, impl):
    samples = [r[(impl, index)] for r in device_rounds if (impl, index) in r]
    if not samples:
        return None
    return {
        "ns_rounds": [s["ns"] for s in samples],
        "ns": stats.median([s["ns"] for s in samples]),
        "ops": samples[0]["ops"],
    }


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


def measure_noise(control_before, control_after, control_ids):
    changes = {}
    for call_id in control_ids:
        base = series(control_before, call_id, "default")
        other = series(control_after, call_id, "default")
        if base["status"] == "ok" and other["status"] == "ok":
            changes[call_id] = stats.per_round_changes(base["us"], other["us"])
    return {
        "calls": control_ids,
        "per_call_changes_pct": changes,
        "noise_pct": stats.noise_pct(changes),
    }


def build(calls_doc, raw, meta):
    """`raw`: time runs per label (`before`, `after`, `ctl_before`, `ctl_after`: one worker JSON or
    None per round), `device`: per round `{(impl, call index): {"ns", "ops"}}`, `control_ids`."""
    noise = measure_noise(raw["time"]["ctl_before"], raw["time"]["ctl_after"], raw["control_ids"])
    noise_pct = noise["noise_pct"]
    rows, skipped = [], []
    for index, call in enumerate(calls_doc["calls"]):
        cid = call["id"]
        before = {"default": series(raw["time"]["before"], cid, "default")}
        after = {impl: series(raw["time"]["after"], cid, impl) for impl in ("default", "native", "codegen")}
        q1 = _comparison(before["default"], after["default"], noise_pct)
        q2 = {impl: _comparison(after["default"], after[impl], noise_pct) for impl in ("native", "codegen")}
        beaten_by = [i for i, c in q2.items() if c["comparable"] and c["verdict"] == "faster"]
        device = {}
        for impl in ("default", "native", "codegen"):
            found = _device(raw["device"], index, impl)
            if found:
                device[impl] = found
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
                "q1": q1,
                "q2": q2,
                "fastest_option_used": None if not any(c["comparable"] for c in q2.values()) else not beaten_by,
                "beaten_by": beaten_by,
                "device": device,
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
    q2 = {"default fastest": 0, "beaten": 0, "not comparable": 0}
    for row in rows:
        used = row["fastest_option_used"]
        q2["not comparable" if used is None else "default fastest" if used else "beaten"] += 1
    return {"q1": q1, "q1_geomean_pct": stats.geomean_change_pct(changes), "q2": q2}
