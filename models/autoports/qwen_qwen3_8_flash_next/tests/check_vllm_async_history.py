# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate async state using equal decode-width histories, retaining token IDs.

This diagnostic does not replace mixed arrivals, chunked prefill, or quality
gates. Pause only before admission so both requests start together; never pause
or drain between their decode steps. The server's events must prove B2 decode.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from models.autoports.qwen_qwen3_8_flash_next.tests.check_vllm_mixed import events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--prompt-lengths", default="65,67")
    parser.add_argument("--output-tokens", type=int, default=96)
    args = parser.parse_args()
    lengths = tuple(map(int, args.prompt_lengths.split(",")))
    if len(lengths) != 2 or min(lengths) < 1 or args.output_tokens < 65:
        parser.error("Supply two positive prompt lengths and at least 65 output tokens")
    if args.output.exists():
        parser.error("Use a fresh output path")
    url = args.server_url
    model = requests.get(url + "/v1/models", timeout=30).json()["data"][0]["id"]
    prompts = [
        (pattern * ((length + 2) // 3))[:length]
        for pattern, length in zip(([785, 13458, 13], [785, 3432, 13]), lengths, strict=True)
    ]
    results = []
    report = dict(
        passed=False,
        scope="served_equal_width_async_diagnostic",
        results=results,
        prompts=prompts,
        output_tokens=args.output_tokens,
        async_scheduling=args.async_scheduling,
    )
    log_offset = len(args.server_log.read_text())

    def post(suffix, **kwargs):
        response = requests.post(url + suffix, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()

    try:
        for order in ((0, 1), (1, 0), (0, 1)):
            offset = len(events(args.server_log))
            post("/pause?mode=keep&clear_cache=false")
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(
                        post,
                        "/v1/completions",
                        json=dict(
                            model=model,
                            prompt=[prompts[i] for i in order],
                            temperature=0,
                            top_k=1,
                            max_tokens=args.output_tokens,
                            ignore_eos=True,
                            return_token_ids=True,
                        ),
                    )
                    # API admission is asynchronous. The trace evidence below,
                    # rather than this delay, proves actual equal-width work.
                    time.sleep(1)
                    if pending.done():
                        raise RuntimeError("Paused request completed before resume")
                    post("/resume")
                    response = pending.result(timeout=300)
            finally:
                post("/resume")
            time.sleep(0.2)
            current = events(args.server_log)[offset:]
            decodes = [e for e in current if e["kind"] == "decode_submission"]
            tokens = [None, None]
            for choice in response["choices"]:
                tokens[order[choice["index"]]] = choice.get("token_ids")
            results.append(dict(order=order, tokens=tokens, response=response, events=current))
            assert all(
                t is not None and len(t) == args.output_tokens for t in tokens
            ), "Missing complete token-ID output"
            assert len(decodes) >= args.output_tokens - 1, "Decode history is incomplete"
            assert all(e["active_slots"] == [0, 1] for e in decodes), "Physical decode width was not always 2"
            history = [e["positions"] for e in decodes]
            results[-1]["position_history"] = history
            assert all(
                after == [position + 1 for position in before] for before, after in zip(history, history[1:])
            ), "Resident positions did not advance exactly once"
            boundaries = [
                list(range(32 * (length // 32 + 1), length + args.output_tokens - 1, 32)) for length in lengths
            ]
            assert all(len(values) >= 2 for values in boundaries), "Need two page boundaries per slot"
            assert all(any(p[slot] == boundary for p in history) for slot in (0, 1) for boundary in boundaries[slot])
            refreshes = [e for e in decodes if e["reload_page_table"] and not e["reload_inputs"]]
            results[-1]["page_only_refreshes"] = len(refreshes)
            assert refreshes and all(e["submitted_page_table"] is not None for e in refreshes)
            tables = [e["submitted_page_table"] for e in decodes if e["submitted_page_table"] is not None]
            assert any(a != b for a, b in zip(tables, tables[1:])), "No actual page-table mapping growth"
            assert all(e.get("sampling_mode", "device") == "device" for e in current), "Unexpected host path"
            assert all(e.get("out_of_vocabulary", 0) == 0 for e in current), "OOV output"
        report["repeat_and_row_order_equal"] = all(r["tokens"] == results[0]["tokens"] for r in results)
        if args.control:
            control = json.loads(args.control.read_text())
            assert control["passed"] and not control["async_scheduling"], "Invalid synchronous control"
            assert control["prompts"] == prompts and control["output_tokens"] == args.output_tokens
            assert all(
                current["order"] == tuple(previous["order"])
                and current["position_history"] == previous["position_history"]
                for current, previous in zip(results, control["results"], strict=True)
            ), "Control execution histories differ"
            report["control_equal"] = [
                current["tokens"] == previous["tokens"]
                for current, previous in zip(results, control["results"], strict=True)
            ]
        log = args.server_log.read_text()[log_offset:]
        overlap = [
            int(n) for n in re.findall(r"TT submissions: \d+ ordinary decode, \d+ verify, (\d+) overlapped", log)
        ]
        unsafe = [
            int(n)
            for n in re.findall(
                r"TT async decode: \d+ submission\(s\) overlapped an outstanding step, (\d+) of them", log
            )
        ]
        report["overlap_counters"] = overlap
        report["unsafe_overlap_counters"] = unsafe
        if args.async_scheduling:
            assert overlap and max(overlap) > 0 and unsafe and max(unsafe) == 0, "Missing safe overlap evidence"
        report["passed"] = report["repeat_and_row_order_equal"] and all(report.get("control_equal", [True]))
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "prompts")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
