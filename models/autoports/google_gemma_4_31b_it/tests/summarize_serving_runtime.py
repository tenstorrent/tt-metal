# SPDX-License-Identifier: Apache-2.0
"""Summarize a bounded server-log interval without opening TT devices."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .check_served_state import log_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-byte", type=int, default=0)
    parser.add_argument("--end-byte", type=int)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    raw = args.server_log.read_bytes()
    end = len(raw) if args.end_byte is None else args.end_byte
    if not 0 <= args.start_byte <= end <= len(raw):
        raise ValueError("Invalid log interval")
    events = log_events(raw[args.start_byte : end])
    sampling = [event for event in events if event.get("event") == "sampling_mode"]
    decodes = [event for event in sampling if event.get("phase") == "decode"]
    root = Path(__file__).resolve().parents[1]
    report = {
        "scope": args.scope,
        "server_log": str(args.server_log),
        "log_byte_range": [args.start_byte, end],
        "sampling_steps": dict(Counter(f'{e.get("phase")}:{e.get("mode")}' for e in sampling)),
        "decode_page_only_refreshes": sum(bool(e.get("reload_page_table")) for e in decodes),
        "decode_full_reloads": sum(bool(e.get("reload_inputs")) for e in decodes),
        "decode_unchanged_steps": sum(not e.get("reload_inputs") and not e.get("reload_page_table") for e in decodes),
        "batched_prefill_groups": [event for event in events if event.get("event") == "generator_batched_prefill"],
        "preparation": [event for event in log_events(raw[:end]) if event.get("event") == "generator_trace_prepare"],
        "registration": [event for event in log_events(raw[:end]) if event.get("event") == "autoport_registration"],
        "source_hashes_at_audit": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "tt").glob("*.py"))
        },
        "limitations": [
            "Counts cover the named log interval, including any benchmark warmup/test requests in it.",
            "Source hashes are audit-time state; launch provenance must establish that the loaded source matches.",
        ],
    }
    if not sampling:
        raise ValueError("No sampling events in interval; missing counters are not zero")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"sampling_steps": report["sampling_steps"], "batched_groups": len(report["batched_prefill_groups"])}
        )
    )


if __name__ == "__main__":
    main()
