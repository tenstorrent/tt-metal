# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare native allocator-driven async HTTP decode against a saved sync run.

This client imports no TTNN and never changes tokens, positions or page tables
inside the server. Reduced-layer text is a numerical diagnostic, not quality
evidence. The control and candidate both use selected real layers 0 and 3.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI


async def check(args):
    control = json.loads(args.control.read_text())
    baseline = [json.loads(line) for line in args.events.read_text().splitlines()]
    loaded = next(event for event in reversed(baseline) if event["event"] == "adapter_loaded")
    assert loaded["layers"] == [0, 3]
    results = []
    async with AsyncOpenAI(base_url=args.server_url.rstrip("/") + "/v1", api_key="empty") as client:

        async def send(index, phase):
            response = await client.completions.create(
                model="meta-models/Muse-Glimmer-30B",
                prompt=control["prompts"][index],
                max_tokens=100,
                temperature=0,
                extra_body={"ignore_eos": True, "return_token_ids": True},
            )
            raw = response.model_dump()
            actual = raw["choices"][0]["token_ids"]
            expected = control["control"][index]["choices"][0]["token_ids"]
            results.append(
                dict(
                    index=index,
                    phase=phase,
                    response=raw,
                    exact_sync_match=actual == expected,
                    crossed_page_boundaries=[
                        position
                        for position in range(len(control["prompts"][index]), len(control["prompts"][index]) + 99)
                        if position % 32 == 0
                    ],
                )
            )

        for index in range(3):
            await send(index, "sequential_async")
        for order in ([0, 1, 2], [2, 0, 1]):
            await asyncio.gather(*(send(index, f"concurrent_order_{order}") for index in order))

    events = [json.loads(line) for line in args.events.read_text().splitlines()][len(baseline) :]
    submits = [event for event in events if event["event"] == "decode_submit"]
    page_only = [event for event in submits if event["reload_page_table"] and not event["reload_inputs"]]
    resident = [event for event in submits if not event["reload_page_table"] and not event["reload_inputs"]]
    overlaps = [
        int(value) for value in re.findall(r"(\d+) overlapped an outstanding step", args.server_log.read_text())
    ]
    failures = []
    if len(results) != 9 or not all(row["exact_sync_match"] for row in results):
        failures.append("Async sequential/reordered completions differ from synchronous token control")
    if not all(row["response"]["usage"]["completion_tokens"] == 100 for row in results):
        failures.append("A completion did not reach the unchanged 100-token cap")
    if not all(len(row["crossed_page_boundaries"]) >= 3 for row in results):
        failures.append("A request did not cross three native page boundaries")
    if len(page_only) < 9 or not resident:
        failures.append("Insufficient page-only refresh or resident steady-decode evidence")
    if not overlaps or max(overlaps) == 0:
        failures.append("Shared plugin reports no overlapped submissions")
    if any(not event["sample"] for event in submits) or any(
        event["event"] in ("host_sampling_compatibility", "capture", "teardown_release") for event in events
    ):
        failures.append("Unexpected sampling fallback or trace lifecycle change during requests")
    evidence = dict(
        status="pass" if not failures else "fail",
        failures=failures,
        control=str(args.control),
        candidate_server=loaded,
        workload=dict(prompt_lengths=[129, 130, 131], output_tokens=100, max_num_seqs=4, native_page_size=32),
        request_results=results,
        decode_submits=len(submits),
        page_only_submits=len(page_only),
        resident_unchanged_submits=len(resident),
        reported_overlapped_submissions=overlaps,
        page_only_events=page_only,
        events_path=str(args.events),
        server_log=str(args.server_log),
        caveat="Reduced real-layer numerical control; generated text is not qualitative evidence.",
    )
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in evidence.items()
                if key not in ("request_results", "candidate_server", "page_only_events")
            },
            indent=2,
        )
    )
    assert not failures, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(check(parser.parse_args()))
