# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize generator serving events in an explicitly bounded log interval."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(path, start_byte=0):
    routes, reloads, kinds, captures = Counter(), Counter(), Counter(), Counter()
    oov = 0
    with path.open("rb") as handle:
        handle.seek(start_byte)
        for raw in handle:
            marker = b"QWEN_VLLM_EVENT "
            if marker not in raw:
                continue
            event = json.loads(raw.split(marker, 1)[1])
            kind = event["kind"]
            kinds[kind] += 1
            if kind in ("decode_submission", "prefill_submission"):
                routes[kind + ":" + event["sampling_mode"]] += 1
            if kind == "decode_submission":
                reloads[
                    "inputs" if event["reload_inputs"] else "page_only" if event["reload_page_table"] else "unchanged"
                ] += 1
            if kind == "capture":
                captures[json.dumps(event["signature"])] += 1
            if kind == "sampled_tokens":
                oov += event["out_of_vocabulary"]
        end_byte = handle.tell()
    return dict(
        source=str(path),
        start_byte=start_byte,
        end_byte=end_byte,
        routes=dict(routes),
        reloads=dict(reloads),
        event_counts=dict(kinds),
        capture_counts=dict(captures),
        host_submissions=sum(count for route, count in routes.items() if route.endswith(":host")),
        device_submissions=sum(count for route, count in routes.items() if route.endswith(":device")),
        out_of_vocabulary=oov,
        scope="Log interval only; whole-lifetime trace reuse requires mixed-shape baseline plus workload events.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--start-byte", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-device-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.log, args.start_byte)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.require_device_only:
        assert result["device_submissions"] > 0, "Missing device route evidence"
        assert result["host_submissions"] == result["out_of_vocabulary"] == 0, result
    print(json.dumps({key: result[key] for key in ("routes", "reloads", "out_of_vocabulary")}))


if __name__ == "__main__":
    main()
