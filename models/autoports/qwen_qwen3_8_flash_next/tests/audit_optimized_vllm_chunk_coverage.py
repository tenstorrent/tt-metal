# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check completed sweep logs for continuous decode and retained prefill traces.

These event counts supplement the official client ITL metric; no server event
interval is substituted for a client token receipt interval.
"""

import argparse
import hashlib
import json
from pathlib import Path


def audit(path, budget):
    data = path.read_bytes()
    events = [
        json.loads(line.split("QWEN_VLLM_EVENT ", 1)[1])
        for line in data.decode().splitlines()
        if "QWEN_VLLM_EVENT " in line
    ]
    groups, current = [], None
    for event in events:
        if event["kind"] == "prefill_submission" and event["starts"] == [0]:
            assert current is None, "Unexpected overlapping prefill groups"
            current = []
        if current is not None:
            current.append(event)
            if event["kind"] == "prefill_result" and any(event["sample_mask"]):
                groups.append(current)
                current = None
    assert current is None, "Incomplete final prefill group"
    rows = []
    for group in groups:
        prefill = [event for event in group if event["kind"] == "prefill_submission"]
        length = prefill[-1]["ends"][0]
        if length not in (4096, 32768):
            continue
        assert all(len(event["starts"]) == len(event["ends"]) == len(event["slots"]) == 1 for event in prefill)
        assert all(event["slots"] == prefill[0]["slots"] for event in prefill)
        assert all(a["ends"] == b["starts"] for a, b in zip(prefill, prefill[1:]))
        assert all(event["ends"][0] > event["starts"][0] for event in prefill)
        decodes = [event for event in group if event["kind"] == "decode_submission"]
        captures = sum(event["kind"] == "capture" for event in group)
        retirements = sum("release" in event["kind"] or "retir" in event["kind"] for event in group)
        active = all(event["active_decode_slots"] for event in prefill)
        host = sum(event.get("sampling_mode") == "host" for event in group)
        oov = sum(event.get("out_of_vocabulary", 0) for event in group)
        row = dict(
            budget=budget,
            prompt_tokens=length,
            prefill_calls=len(prefill),
            physical_chunk_lengths=sorted({event["ends"][0] - event["starts"][0] for event in prefill}),
            all_prefill_calls_have_active_decoder=active,
            decode_submissions=len(decodes),
            captures=captures,
            retirements=retirements,
            host_routes=host,
            out_of_vocabulary=oov,
            server_window_seconds=group[-1]["time"] - group[0]["time"],
            server_log=str(path),
            server_log_sha256=hashlib.sha256(data).hexdigest(),
        )
        row["passed"] = active and bool(decodes) and not (captures or retirements or host or oov)
        rows.append(row)
    assert sorted(row["prompt_tokens"] for row in rows) == [4096, 32768], "Need exactly two completed windows"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for run in manifest["runs"]:
        assert run.get("devices_free") and run.get("remaining_owned_pids") == [], "Sweep arm is not closed"
        assert not run["result"]["errors"]
        rows.extend(audit(Path(run["evidence"]) / "server.log", run["budget"]))
    result = dict(
        passed=bool(rows) and all(row["passed"] for row in rows),
        scope="Completed sweep server-event coverage; supplements official client ITL, not publication timing.",
        manifest=str(args.manifest),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], windows=len(rows), output=str(args.output))))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
