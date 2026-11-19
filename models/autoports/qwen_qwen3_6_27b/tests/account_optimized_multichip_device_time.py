# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, fail-conservative stage5 device-occupation ledger.

Stream individual command logs. Count unions of timestamped occupation
intervals, never four times TP4 elapsed time. Unknown endpoints stay unknown;
wrapper tails/profiler postprocessing cannot create extra device time.
"""

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d(?:\.\d+)?)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EVENTS = (
    ("open_driver", "Opening user mode device driver"),
    ("cluster_ready", "Starting devices in cluster completed"),
    ("fabric_begin", "Initializing Fabric ("),
    ("fabric_ready", "Fabric initialized on "),
    ("close_begin", "Closing user mode device drivers"),
    ("close_cluster_begin", "Closing devices in cluster ("),
    ("close_complete", "Closing devices in cluster completed"),
)


def parse_time(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError:
        return None


def iso(value):
    return value.isoformat() if value else None


def elapsed(start, end):
    return (end - start).total_seconds() if start and end and end >= start else None


def interval_union(intervals):
    """Union intervals globally: nested/duplicate/profile logs never double count."""
    ordered = sorted((parse_time(x["start_utc"]), parse_time(x["end_utc"]), x["ref"]) for x in intervals)
    groups = []
    for start, end, ref in ordered:
        if not start or not end or end < start:
            raise ValueError("Invalid interval passed to union")
        if groups and start <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(ref)
        else:
            groups.append([start, end, [ref]])
    return [{"start_utc": iso(a), "end_utc": iso(b), "seconds": elapsed(a, b), "refs": refs} for a, b, refs in groups]


def option(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def command_key(argv, repo):
    result = list(argv)
    for flag in ("--output", "--reference", "--activation-file", "--policy-file"):
        value = option(result, flag)
        if value:
            p = Path(value)
            result[result.index(flag) + 1] = str(p.resolve() if p.is_absolute() else (repo / p).resolve())
    value = option(result, "--policy-json")
    if value:
        try:
            result[result.index("--policy-json") + 1] = json.dumps(json.loads(value), sort_keys=True)
        except ValueError:
            pass
    return tuple(result)


def decoder_sha(record):
    s = record.get("source_sha256")
    return next((v for k, v in s.items() if k.endswith("tt/multichip_decoder.py")), None) if isinstance(s, dict) else s


def scan_log(path, display, cutoff=None):
    events, first, last, bytes_read = [], None, None, 0
    # Read only the size present at snapshot start, so an active job cannot
    # extend the scan indefinitely. No mtime is treated as an execution time.
    size = path.stat().st_size
    with path.open("rb") as stream:
        line_number = 0
        continuation = False
        while stream.tell() < size:
            line = stream.readline(min(1024 * 1024, size - stream.tell()))
            if not line:
                break
            bytes_read += len(line)
            was_continuation = continuation
            continuation = not line.endswith(b"\n")
            if was_continuation:
                continue
            line_number += 1
            text = ANSI.sub("", line[:1024].decode(errors="replace"))
            match = STAMP.match(text)
            if not match:
                continue
            stamp = parse_time(match[1] + "T" + match[2] + "+00:00")
            if cutoff and stamp > cutoff:
                continue
            first = min(first, stamp) if first else stamp
            last = max(last, stamp) if last else stamp
            for kind, token in EVENTS:
                if token in text:
                    event = {
                        "event": kind,
                        "utc": iso(stamp),
                        "line": line_number,
                        "evidence": f"{display}:{line_number}",
                        "message": text.strip()[:400],
                    }
                    if kind == "fabric_ready":
                        count = re.search(r"Fabric initialized on (\d+) devices", text)
                        event["devices"] = int(count[1]) if count else None
                    events.append(event)
                    break
    return {
        "first_timestamp_utc": iso(first),
        "last_timestamp_utc": iso(last),
        "bytes_scanned": bytes_read,
        "snapshot_size_bytes": size,
        "events": events,
    }


def sessions(events):
    """Associate explicit open/ready/close markers; unmatched events stay visible."""
    result, current, unmatched = [], None, []
    for event in events:
        kind = event["event"]
        if kind == "open_driver":
            if current:
                current["status"] = "missing_close_before_next_open"
                result.append(current)
            current = {"events": [event], "status": "open_without_close"}
        elif current:
            current["events"].append(event)
            if kind == "close_complete":
                current["status"] = "complete"
                result.append(current)
                current = None
        elif kind in ("cluster_ready", "fabric_begin", "fabric_ready"):
            current = {"events": [event], "status": "ready_without_open_or_close"}
        else:
            unmatched.append(event)
    if current:
        result.append(current)
    for session in result:
        markers = session["events"]
        get = lambda kinds: next((x for x in markers if x["event"] in kinds), None)
        opened = get(("open_driver",))
        ready = get(("cluster_ready", "fabric_ready"))
        # Use the first observed teardown-start marker as the strict floor's
        # endpoint. Teardown itself remains outside that conservative count.
        closing = get(("close_begin", "close_cluster_begin"))
        closed = get(("close_complete",))

        def make(a, b):
            seconds = elapsed(parse_time(a["utc"]), parse_time(b["utc"])) if a and b else None
            return (
                {
                    "start_utc": a["utc"],
                    "end_utc": b["utc"],
                    "seconds": seconds,
                    "start_evidence": a["evidence"],
                    "end_evidence": b["evidence"],
                }
                if seconds is not None
                else None
            )

        session["confirmed_occupation_floor"] = make(ready, closing)
        session["observed_open_to_close"] = make(opened, closed)
        session["device_count"] = next((x.get("devices") for x in markers if x["event"] == "fabric_ready"), None)
        session["unknown_endpoint"] = not session["confirmed_occupation_floor"] or not session["observed_open_to_close"]
    return result, unmatched


def build(directory, repo, snapshot):
    display = lambda p: str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p)
    rows, warnings, excluded = [], [], []
    ledger = []
    ledger_path = directory / "candidate_commands.jsonl"
    if ledger_path.exists():
        for index, line in enumerate(ledger_path.open(), 1):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get("command"), list):
                    ledger.append((obj, f"{display(ledger_path)}:{index}"))
            except ValueError:
                warnings.append(f"{display(ledger_path)}:{index}: malformed/incomplete JSON")
    used_ledger, seen_logs = set(), set()
    for command_path in sorted(directory.rglob("*.command.json")):
        try:
            command = json.loads(command_path.read_text())
        except (OSError, ValueError) as exc:
            warnings.append(f"{display(command_path)}: {exc}")
            rows.append({"id": display(command_path), "status": "unreadable_command", "sessions": []})
            continue
        if not isinstance(command, dict) or not isinstance(command.get("command"), list):
            warnings.append(f"{display(command_path)}: missing argv")
            rows.append({"id": display(command_path), "status": "invalid_command_missing_argv", "sessions": []})
            continue
        argv, stamp = command["command"], parse_time(command.get("utc"))
        path = command_path.with_name(command_path.name.removesuffix(".command.json") + ".log")
        row = {
            "id": display(command_path),
            "command_path": display(command_path),
            "command": argv,
            "command_utc": iso(stamp),
            "log_path": display(path),
            "sessions": [],
            "ledger_ref": None,
            "recorded_process_wall_seconds": None,
            "regime": command.get("regime", "unspecified"),
            "source_decoder_sha256": decoder_sha(command),
        }
        matches = []
        if stamp:
            for i, (record, ref) in enumerate(ledger):
                other = parse_time(record.get("utc"))
                if i in used_ledger or not other or abs((stamp - other).total_seconds()) > 30:
                    continue
                if command_key(argv, repo) == command_key(record["command"], repo) and decoder_sha(
                    record
                ) == decoder_sha(command):
                    matches.append((abs((stamp - other).total_seconds()), i))
        if matches:
            i = min(matches)[1]
            used_ledger.add(i)
            record, ref = ledger[i]
            duration = record.get("elapsed_seconds")
            row.update(
                ledger_ref=ref,
                exit_code=record.get("exit_code"),
                regime=record.get("regime", "unspecified"),
                recorded_process_wall_seconds=duration if type(duration) in (int, float) and duration >= 0 else None,
                process_wall_start_utc=record.get("utc"),
                chips=record.get("chips"),
            )
        if path.is_file():
            seen_logs.add(path)
            scanned = scan_log(path, display(path), snapshot)
            row.update(scanned)
            row["sessions"], row["unmatched_events"] = sessions(scanned["events"])
            row["status"] = "has_device_intervals" if row["sessions"] else "no_device_interval_markers"
            first = parse_time(scanned["first_timestamp_utc"])
            row["command_log_binding"] = (
                "consistent" if stamp and first and first >= stamp else "unknown_or_log_predates_command"
            )
            row["observed_command_to_last_log_seconds"] = elapsed(stamp, parse_time(scanned["last_timestamp_utc"]))
            if row["command_log_binding"] != "consistent":
                row["binding_note"] = (
                    "Device event timestamps remain evidence; command identity is not inferred across a reused/missing sidecar"
                )
            del row["events"]
        else:
            row["status"] = "missing_log"
        rows.append(row)
    # Every unpaired log remains named. Driver/watcher/profiler logs often
    # repeat child tails; they never independently contribute occupation time.
    for path in sorted(directory.rglob("*.log")):
        if path in seen_logs:
            continue
        excluded.append(
            {
                "path": display(path),
                "reason": "no individual command sidecar; driver/watcher/profile/CPU wrapper or unknown; not counted",
            }
        )
    for i, (record, ref) in enumerate(ledger):
        if i not in used_ledger:
            rows.append(
                {
                    "id": ref,
                    "status": "historical_execution_without_matched_log",
                    "command": record["command"],
                    "command_utc": record.get("utc"),
                    "ledger_ref": ref,
                    "recorded_process_wall_seconds": record.get("elapsed_seconds"),
                    "exit_code": record.get("exit_code"),
                    "sessions": [],
                    "note": "A reused output/sidecar must not donate current device timestamps to this execution",
                }
            )
    floors, observed = [], []
    exact_seen = {}
    for row in rows:
        for index, session in enumerate(row["sessions"]):
            ref = f"{row['id']}#session{index}"
            markers = tuple((e["event"], e["utc"]) for e in session["events"])
            if markers in exact_seen:
                session["duplicate_of"] = exact_seen[markers]
            else:
                exact_seen[markers] = ref
            for key, target in (("confirmed_occupation_floor", floors), ("observed_open_to_close", observed)):
                if session[key]:
                    target.append({**session[key], "ref": ref})
    floor_union, open_union = interval_union(floors), interval_union(observed)
    floor_seconds = math.fsum(x["seconds"] for x in floor_union)
    starts = [parse_time(r.get("command_utc")) for r in rows]
    starts = [x for x in starts if x]
    ends = [parse_time(r.get("last_timestamp_utc")) for r in rows]
    ends = [x for x in ends if x]
    return {
        "schema_version": 1,
        "stage": "optimized_multichip_decoder",
        "snapshot_utc": iso(snapshot),
        "script_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timestamp_timezone": "UTC; command sidecars explicit UTC; native log timestamps interpreted in this run's UTC environment",
        "unit": "occupied-device wall seconds; TP4 counts once, never chip-seconds",
        "method": {
            "conservative_floor": "union of cluster-ready (or fabric-ready if cluster marker missing) through first teardown-start; both endpoints required",
            "observed_envelope": "union of explicit driver-open through completed cluster close; includes initialization/teardown",
            "wall_time": "process elapsed is supplementary only; command-to-last-log is an observed wall span, not process duration or device time",
            "deduplication": "exact event-sequence duplicates are linked; global interval union removes all nested/overlapping wrapper/profile time",
            "unknowns": "missing logs/endpoints/unmatched historical attempts are preserved and contribute no guessed seconds",
            "artifacts": "No individual artifact hashing, compression or archiving; only this source file is hashed",
        },
        "summary": {
            "command_or_historical_records": len(rows),
            "session_count": sum(len(r["sessions"]) for r in rows),
            "conservative_occupied_seconds": floor_seconds,
            "conservative_occupied_hours": floor_seconds / 3600,
            "observed_open_to_close_union_seconds": math.fsum(x["seconds"] for x in open_union),
            "raw_observed_interval_seconds_before_union": math.fsum(x["seconds"] for x in observed),
            "raw_floor_seconds_before_union": math.fsum(x["seconds"] for x in floors),
            "records_without_sessions": sum(not r["sessions"] for r in rows),
            "sessions_with_unknown_endpoint": sum(s["unknown_endpoint"] for r in rows for s in r["sessions"]),
            "unmatched_close_event_count": sum(len(r.get("unmatched_events", [])) for r in rows),
            "excluded_logs": len(excluded),
            "status_counts": dict(collections.Counter(r["status"] for r in rows)),
            "first_command_utc": iso(min(starts)) if starts else None,
            "latest_timestamped_log_utc": iso(max(ends)) if ends else None,
            "observed_activity_wall_span_seconds": elapsed(min(starts), max(ends)) if starts and ends else None,
            "first_command_to_snapshot_wall_seconds": elapsed(min(starts), snapshot) if starts else None,
        },
        "stop_rule": {
            "rule_source": "/home/ttuser/qwen36-27b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt",
            "required_expression": "(roofline_fraction >= 0.70 AND each of last two kept improvements < 0.03) OR accumulated device time >= 28800 seconds",
            "threshold_device_seconds": 28800,
            "roofline_fraction": None,
            "last_two_kept_improvements": None,
            "verdict": "not evaluated; no stop-rule firing or completion claim",
            "note": "Original user rule overrides weaker skill defaults. Missing intervals do not prove zero time; wall elapsed is not device time. Quality gates remain required.",
        },
        "conservative_union": floor_union,
        "observed_union": open_union,
        "records": rows,
        "excluded_logs": excluded,
        "warnings": warnings,
    }


def markdown(report):
    s = report["summary"]
    wall_span = (
        "unknown"
        if s["observed_activity_wall_span_seconds"] is None
        else f"{s['observed_activity_wall_span_seconds']:.3f} seconds"
    )
    wall_snapshot = (
        "unknown"
        if s["first_command_to_snapshot_wall_seconds"] is None
        else f"{s['first_command_to_snapshot_wall_seconds']:.3f} seconds"
    )
    lines = [
        "# Stage5 device-time ledger",
        "",
        f"Snapshot: {report['snapshot_utc']}. Source: `device_time_ledger.json`.",
        "",
        f"Conservative confirmed occupation: **{s['conservative_occupied_seconds']:.3f} seconds ({s['conservative_occupied_hours']:.6f} hours)**. TP4 counts once, never four chip-seconds per second.",
        "",
        f"Observed driver-open→completed-close interval union: **{s['observed_open_to_close_union_seconds']:.3f} seconds**. This includes initialization and teardown; the conservative count excludes both. Both require timestamp evidence and exclude unbounded tails.",
        "",
        f"Wall span from first recorded command to latest timestamped log: **{wall_span}**; first command to snapshot: **{wall_snapshot}**. These include host preparation, profiling postprocessing and idle time, and are not device time.",
        "",
        "The strict user stop rule is `(>=70% roofline AND last two kept improvements each <3%) OR 8 hours accumulated device time`. This ledger does not evaluate the roofline/kept-improvement branch and makes **no stop-rule firing or completion claim**. Quality gates remain required.",
        "",
        "Native log timestamps are interpreted as UTC, matching this run's environment and UTC command sidecars. A complete interval begins at confirmed cluster readiness (or fabric readiness if needed) and ends at the first teardown marker. Driver open/close is also preserved separately. No file mtime, timeout limit, current time, reported process duration or profiler kernel sum is substituted for a missing endpoint.",
        "",
        f"Records: {s['command_or_historical_records']}; device sessions: {s['session_count']}; sessions with an unknown endpoint: {s['sessions_with_unknown_endpoint']}; records without sessions: {s['records_without_sessions']}; unmatched close markers: {s['unmatched_close_event_count']}; excluded wrapper/sidecar-less logs: {s['excluded_logs']}.",
        "",
        "Individual command logs are streamed. Exact repeated event sequences retain duplicate links; the global interval union prevents duplicate/nested/profile-wrapper time from being added twice. Archived attempts remain separate from newer executions with reused output names. Every evidence path/line, interval, command and unknown record is preserved in JSON. No individual artifact was hashed or archived.",
        "",
        "## Incomplete records",
        "",
        "| Record | Status / limitation |",
        "|---|---|",
    ]
    for r in report["records"]:
        notes = []
        if not r["sessions"]:
            notes.append(r["status"])
        notes += [f"session{i}: {v['status']}" for i, v in enumerate(r["sessions"]) if v["unknown_endpoint"]]
        if r.get("unmatched_events"):
            notes.append("unmatched close markers")
        if notes:
            lines.append(f"| `{r['id']}` | {'; '.join(notes)} |")
    if not any(
        not r["sessions"] or r.get("unmatched_events") or any(v["unknown_endpoint"] for v in r["sessions"])
        for r in report["records"]
    ):
        lines.append("| — | No missing endpoints among inspected command logs; excluded logs remain outside scope. |")
    lines += [
        "",
        "Regenerate:",
        "",
        "```bash",
        "python models/autoports/qwen_qwen3_6_27b/tests/account_optimized_multichip_device_time.py",
        "```",
        "",
    ]
    if report["warnings"]:
        lines += ["Warnings:", ""] + ["- " + x for x in report["warnings"]] + [""]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=root / "doc/optimized_multichip_decoder")
    p.add_argument("--output", type=Path)
    p.add_argument("--markdown", type=Path)
    args = p.parse_args()
    report = build(args.directory.resolve(), root.parents[2], datetime.now(timezone.utc))
    output = args.output or args.directory / "device_time_ledger.json"
    md = args.markdown or args.directory / "device_time_ledger.md"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    md.write_text(markdown(report))
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "output": str(output),
                "markdown": str(md),
                "warnings": len(report["warnings"]),
            }
        )
    )


if __name__ == "__main__":
    main()
