# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, reproducible ledger of completed native device-session intervals.

Use driver opening/closing timestamps, not file mtimes or first/last candidate
wall time. Overlapping/duplicated logs are unioned, never summed twice. This is
device ownership time (including setup/compilation), NOT summed active kernels.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/fused_decoder"
TIMESTAMP = re.compile(r"(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
OPEN = "Opening user mode device driver"
CLOSE = "Closing user mode device drivers"


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def iso(value):
    return value.isoformat()


def transcript_reason(text, relative):
    """Quoted tool output cannot establish a continuous native driver session.

    Even matching transcript pairs are excluded: their direct job logs must
    supply the ownership evidence. Do not join excerpts from separate jobs.
    """
    name = Path(relative).name.lower()
    if "launcher" in name or "queue" in name:
        return "launcher/queue transcript; require preserved direct job log"
    if re.search(r"(?m)^OpenAI Codex v|^session id:", text):
        return "agent transcript header; quoted native markers are not job evidence"
    if re.search(r"(?m)^--- last \d+ lines of .+ ---$", text):
        return "wrapper tail excerpts; require preserved direct job log"
    return None


def parse_log(path, relative, after):
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    reason = transcript_reason(text, relative)
    if reason:
        markers = sum(OPEN in line or CLOSE in line for line in text.splitlines())
        return (
            [],
            [dict(log=relative, reason=reason, classification="quoted_transcript", native_marker_count=markers)],
            markers,
            hashlib.sha256(data).hexdigest(),
        )
    pending = None
    ambiguous = False
    sessions, unknown = [], []
    markers = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", raw)
        if OPEN not in line and CLOSE not in line:
            continue
        markers += 1
        found = TIMESTAMP.search(line)
        if not found:
            unknown.append(dict(log=relative, line=line_number, reason="device marker lacks timestamp"))
            ambiguous = True
            continue
        stamp = utc(found.group(1))
        if OPEN in line:
            if pending is not None:
                unknown.append(
                    dict(log=relative, line=line_number, reason="overlapping opens in one log; pairing ambiguous")
                )
                ambiguous = True
            pending = (stamp, line_number)
        elif pending is None:
            if stamp > after:
                unknown.append(
                    dict(log=relative, line=line_number, reason="close without matching open", close_utc=iso(stamp))
                )
        else:
            begin, open_line = pending
            if stamp < begin:
                unknown.append(dict(log=relative, line=line_number, reason="close precedes open"))
            elif not ambiguous and stamp > after:
                start = max(begin, after)
                sessions.append(
                    dict(
                        log=relative,
                        open_line=open_line,
                        close_line=line_number,
                        native_open_utc=iso(begin),
                        start_utc=iso(start),
                        end_utc=iso(stamp),
                        seconds=(stamp - start).total_seconds(),
                        clipped_to_first_candidate=begin < after,
                    )
                )
            pending = None
            ambiguous = False
    if pending is not None:
        unknown.append(
            dict(
                log=relative,
                line=pending[1],
                open_utc=iso(pending[0]),
                reason="open without native close; failed/running/truncated session excluded",
            )
        )
    return sessions, unknown, markers, hashlib.sha256(data).hexdigest()


def union_intervals(sessions):
    merged = []
    for row in sorted(sessions, key=lambda x: x["start_utc"]):
        start, end = utc(row["start_utc"]), utc(row["end_utc"])
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        dict(start_utc=iso(start), end_utc=iso(end), seconds=(end - start).total_seconds()) for start, end in merged
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--after", help="UTC first candidate time; default optimization_progress.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    progress_path = args.doc / "optimization_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    first = args.after or progress.get("first_candidate_measurement_utc")
    if first is None:
        parser.error("--after or optimization_progress.json first_candidate_measurement_utc is required")
    after = utc(first)
    sessions, unknown, hashes, excluded_transcripts = [], [], {}, []
    files = sorted(args.doc.rglob("*.log"))
    no_device_markers = 0
    for path in files:
        relative = str(path.relative_to(args.doc))
        found, issues, markers, digest = parse_log(path, relative, after)
        sessions.extend(found)
        for issue in issues:
            if issue.get("classification") == "quoted_transcript":
                excluded_transcripts.append(dict(**issue, sha256=digest))
            else:
                unknown.append(issue)
        if markers:
            hashes[relative] = digest
        else:
            no_device_markers += 1
    starts = {row["native_open_utc"] for row in sessions}
    ends = {row["end_utc"] for row in sessions}
    for issue in unknown:
        issue["endpoint_found_in_completed_session_elsewhere"] = (
            issue.get("open_utc") in starts or issue.get("close_utc") in ends
        )
    merged = union_intervals(sessions)
    seconds = sum(x["seconds"] for x in merged)
    naive = sum(x["seconds"] for x in sessions)
    result = dict(
        generated_utc=iso(datetime.now(timezone.utc)),
        first_candidate_measurement_utc=iso(after),
        first_candidate_evidence="--after"
        if args.after
        else "optimization_progress.json:first_candidate_measurement_utc",
        timestamp_timezone="UTC; native timestamps have no offset, interpreted using this lane's UTC environment",
        method="Union of complete native driver-open to driver-close intervals, clipped at the first candidate",
        policy=[
            "Includes completed device-owned setup, compilation, transfers, correctness, benchmark and profiler sessions",
            "Excludes CPU-only logs and time outside completed device sessions, including idle gaps between sessions",
            "Closes at the start of native driver teardown; remaining cleanup time is conservatively excluded",
            "Repeated/copied direct job logs are deduplicated by interval union",
            "Launcher/agent/queue transcripts contribute zero; only preserved direct job logs establish paired sessions",
            "Unclosed or ambiguous sessions contribute zero; no inferred endpoint or file mtime is used",
            "CPU computation or idle waits while a device remains open cannot be separated by these markers",
        ],
        log_files_scanned=len(files),
        logs_without_device_markers=no_device_markers,
        completed_session_records=len(sessions),
        union_interval_count=len(merged),
        raw_interval_seconds_before_deduplication=naive,
        overlap_or_duplicate_seconds_removed=naive - seconds,
        cumulative_completed_device_session_seconds=seconds,
        cumulative_completed_device_session_hours=seconds / 3600,
        session_ownership_timebox_hours=8,
        session_ownership_timebox_reached=seconds >= 8 * 3600,
        active_device_kernel_seconds=None,
        stop_rule_met=None,
        stop_rule_note=(
            "This establishes completed session ownership time only. It does not prove active "
            "kernel time or waive correctness/performance/review gates. Session ownership includes "
            "in-session host work; timebox satisfaction needs this policy accepted explicitly. "
            "Elapsed wall time is never substituted for the cumulative ledger."
        ),
        unresolved_marker_records=sum(not issue["endpoint_found_in_completed_session_elsewhere"] for issue in unknown),
        unknown_intervals=unknown,
        excluded_transcript_logs=excluded_transcripts,
        intervals=merged,
        session_evidence=sessions,
        device_log_sha256=hashes,
    )
    output = args.output or args.doc / "device_time_ledger.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "completed_session_records",
                    "union_interval_count",
                    "cumulative_completed_device_session_hours",
                    "session_ownership_timebox_reached",
                    "stop_rule_met",
                )
            },
            indent=2,
        )
    )
    print(f"Unknown intervals: {len(unknown)}; ledger: {output}")


if __name__ == "__main__":
    main()
