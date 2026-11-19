# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict offline join of streaming device durations with host Tracy metadata.

Never imports TTNN. Selection uses the half-open raw dispatcher-ID range saved
by profile_decoder --streaming-profile. Prefill selects its sole execution;
decode selects the final execution of exactly two replay calls (one warmup,
one measured), allowing capture to have executed or only recorded commands.
All IDs must have identical, contiguous execution ordinals and increasing
per-ID cycle intervals. No missing operations, duplicate keys, lost records,
or invented gap/whole-pass device timings are accepted.
"""

import argparse
import csv
import hashlib
import json
import lzma
import math
import re
import subprocess
import sys
from pathlib import Path


def integer(value):
    return int(str(value).strip())


def marker_index(rows, marker):
    matches = [i for i, row in enumerate(rows) if row.get("OP TYPE") == "signpost" and row.get("OP CODE") == marker]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {marker} signpost; found {len(matches)}")
    return matches[0]


def merge_rows(host_rows, device_rows, profile, log):
    if not profile.get("streaming_profile") or not profile.get("runtime_audit") or not profile.get("finite"):
        raise ValueError("Expected a completed, audited streaming-profile run")
    if "[streaming profiler] active" not in log:
        raise ValueError("Streaming profiler activation is not evidenced")
    if "[streaming profiler] capture:" not in log:
        raise ValueError("Streaming final flush/verification is not evidenced")
    for pattern in (
        r"\b[1-9]\d* dropped bytes\b",
        r"\bdropped [1-9]\d*\b",
        r"\b[1-9]\d* order regressions\b",
        r"\b[1-9]\d* id collisions\b",
        r"\b[1-9]\d* foreign metadata sections\b",
        r"\bmissed [1-9]\d* bytes\b",
        r"\b[1-9]\d* words on [1-9]\d* lanes\b",
    ):
        if re.search(pattern, log):
            raise ValueError(f"Incomplete streaming capture: {pattern}")
    mode = profile["mode"]
    selection = profile["streaming_selection"]
    if selection["mode"] != mode or mode not in ("prefill", "decode"):
        raise ValueError("Mode/selection mismatch")
    begin, end = integer(selection["begin"]), integer(selection["end"])
    if not 0 < begin < end < (1 << 21):
        raise ValueError("Invalid/overflowing raw dispatcher-ID range")
    start_marker, end_marker = f"PERF_{mode.upper()}", f"PERF_{mode.upper()}_END"
    measured_start, measured_end = marker_index(host_rows, start_marker), marker_index(host_rows, end_marker)
    if measured_start >= measured_end:
        raise ValueError("Measurement markers are reversed")
    if mode == "decode":
        if selection["trace_replay_calls"] != ["unmeasured_warmup", "measured_final"]:
            raise ValueError("Decode must have exactly one warmup followed by the sole measured final replay")
        template_start, template_end = marker_index(host_rows, "STREAMING_CAPTURE_BEGIN"), marker_index(
            host_rows, "STREAMING_CAPTURE_END"
        )
        if not template_start < template_end < measured_start:
            raise ValueError("Capture markers must precede measured replay")
    else:
        template_start, template_end = measured_start, measured_end
        if selection["trace_replay_calls"]:
            raise ValueError("Eager prefill must not claim trace replays")
    windows = [(template_start, template_end), (measured_start, measured_end)]
    for window_start, window_end in windows:
        for row in host_rows[window_start + 1 : window_end]:
            encoded = row.get("GLOBAL CALL COUNT")
            if encoded not in (None, "") and integer(encoded) >> 31:
                raise ValueError("Host fallback operation in selected window")
    if mode == "decode" and any(
        row.get("OP TYPE") == "tt_dnn_device" for row in host_rows[measured_start + 1 : measured_end]
    ):
        raise ValueError("Measured decode must replay captured programs, not enqueue new device operations")
    expected = set(range(begin, end))
    templates = {}
    order = []
    chips = set()
    for row in host_rows[template_start + 1 : template_end]:
        if row.get("OP TYPE") != "tt_dnn_device":
            continue
        encoded = integer(row["GLOBAL CALL COUNT"])
        if encoded >> 31:
            raise ValueError("Host fallback operation in selected device window")
        raw, chip = encoded >> 10, encoded & 1023
        if raw not in expected or raw in templates:
            raise ValueError(f"Unexpected/duplicate host program ID {raw}")
        templates[raw] = row
        order.append(raw)
        chips.add(chip)
    if set(templates) != expected or order != list(range(begin, end)) or len(chips) != 1:
        raise ValueError("Host metadata must cover every captured raw ID once, in order, on one device")
    chip = chips.pop()
    executions = {raw: {} for raw in expected}
    for row in device_rows:
        raw = integer(row["GLOBAL CALL COUNT"])
        if integer(row["DEVICE ID"]) != chip or raw not in expected:
            continue
        index = integer(row["EXECUTION"])
        if index in executions[raw]:
            raise ValueError(f"Duplicate streaming program/execution {(raw, index)}")
        executions[raw][index] = row
    sets = {tuple(sorted(rows)) for rows in executions.values()}
    if len(sets) != 1:
        raise ValueError("Missing operations or inconsistent execution counts")
    indices = sets.pop()
    permitted = ((0,),) if mode == "prefill" else ((0, 1), (0, 1, 2))
    if indices not in permitted:
        raise ValueError(f"Unexpected execution ordinals {indices}; expected {permitted}")
    chosen = indices[-1]
    intervals = []
    result = [dict(host_rows[measured_start])]
    for raw in order:
        previous_end = None
        for index in indices:
            row = executions[raw][index]
            start, stop = integer(row["DEVICE KERNEL START CYCLE"]), integer(row["DEVICE KERNEL END CYCLE"])
            duration = float(row["DEVICE KERNEL DURATION [ns]"])
            if (
                start <= 0
                or stop <= start
                or not math.isfinite(duration)
                or duration <= 0
                or integer(row["CORE COUNT"]) < 1
            ):
                raise ValueError(f"Invalid streaming timing for {(raw, index)}")
            if previous_end is not None and start <= previous_end:
                raise ValueError(f"Replay order is not strictly increasing for raw ID {raw}")
            previous_end = stop
            intervals.append((raw, index, start, stop))
        merged = dict(templates[raw])
        for name in merged:
            if (name.startswith("DEVICE ") and ("[ns]" in name or "CYCLE" in name)) or name.startswith(
                "OP TO OP LATENCY"
            ):
                merged[name] = ""
            if mode == "decode" and name.startswith("HOST "):
                merged[name] = ""  # Capture host time does not measure replay.
        measured = executions[raw][chosen]
        for name, value in measured.items():
            if name not in ("GLOBAL CALL COUNT", "EXECUTION"):
                merged[name] = value
        merged["STREAMING RAW PROGRAM ID"] = raw
        merged["STREAMING EXECUTION"] = chosen
        merged["OP TO OP LATENCY [ns]"] = ""
        merged["OP TO OP LATENCY BR/NRISC START [ns]"] = ""
        result.append(merged)
    result.append(dict(host_rows[measured_end]))
    proof = {
        "passed": True,
        "mode": mode,
        "raw_id_begin": begin,
        "raw_id_end": end,
        "device_id": chip,
        "expected_ops": end - begin,
        "selected_ops": len(result) - 2,
        "observed_execution_ordinals": list(indices),
        "selected_execution": chosen,
        "selection_rule": "sole eager execution"
        if mode == "prefill"
        else "last execution of captured IDs; exactly one warmup replay followed by measured final replay",
        "markers": [start_marker, end_marker],
        "raw_id_order": order,
        "execution_intervals_cycles": intervals,
        "device_gap_measured": False,
        "whole_pass_device_latency_measured": False,
        "timing_scope": "per-operation device durations; kernel sum, not calibrated whole-pass elapsed time",
        "profiler_stall_summaries": re.findall(r"Device \d+: \d+ profiler stalls[^\n]*", log),
        "profiler_stall_count": sum(int(value) for value in re.findall(r"Device \d+: (\d+) profiler stalls", log)),
        "stall_count_basis": "completed capture verification; device summary is emitted only for nonzero stalls",
    }
    return result, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-csv", type=Path, required=True)
    parser.add_argument("--streaming-csv", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--capture-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.host_csv.open() as stream:
        host_rows = list(csv.DictReader(stream))
    with args.streaming_csv.open() as stream:
        device_rows = list(csv.DictReader(stream))
    profile = json.loads(args.profile_json.read_text())
    merged, proof = merge_rows(host_rows, device_rows, profile, args.capture_log.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "streaming_selected_ops.csv"
    fields = list(dict.fromkeys(key for row in merged for key in row))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    selection_artifact = args.output_dir / "streaming_selection.json.xz"
    selection_artifact.write_bytes(
        lzma.compress(
            json.dumps(
                {
                    "raw_id_order": proof.pop("raw_id_order"),
                    "execution_intervals_cycles": proof.pop("execution_intervals_cycles"),
                },
                separators=(",", ":"),
            ).encode(),
            preset=9,
        )
    )
    proof["selection_artifact"] = {
        "path": str(selection_artifact),
        "sha256": hashlib.sha256(selection_artifact.read_bytes()).hexdigest(),
    }
    proof["inputs"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.host_csv, args.streaming_csv, args.profile_json, args.capture_log)
    }
    proof["profile_sha256"] = hashlib.sha256(args.profile_json.read_bytes()).hexdigest()
    proof["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance = args.output_dir / "streaming_merge.json"
    provenance.write_text(json.dumps(proof, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "models.autoports.qwen_qwen3_8_flash_next.tests.report_performance",
        "--source-csv",
        str(output),
        "--output-dir",
        str(args.output_dir),
        "--profile-json",
        str(args.profile_json),
        "--program-support-count",
        "0",
        "--mode",
        profile["mode"],
        "--streaming-merge-provenance",
        str(provenance),
    ]
    subprocess.run(command, check=True)
    print(
        json.dumps(
            {
                "selected_ops": proof["selected_ops"],
                "selected_execution": proof["selected_execution"],
                "output": str(output),
                "gap_latency": None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
