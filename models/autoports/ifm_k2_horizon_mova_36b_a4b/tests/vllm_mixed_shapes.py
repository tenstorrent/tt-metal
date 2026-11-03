# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-server, first-encounter mixed traffic with stream timing and trace audit."""

import argparse
import concurrent.futures
import json
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--events", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
model = "IFM/K2-Horizon-MoVA-36B-A4B"
tokenizer = AutoTokenizer.from_pretrained(
    model, revision="e9ce717642b2074082875be233c02c729ba47960", trust_remote_code=True, local_files_only=True
)
events_path = Path(a.events)


def events():
    return [json.loads(line) for line in events_path.read_text().splitlines()]


baseline = events()
started = time.time()
first_token = threading.Event()


def run(length, tag, sampled=False, anchor=False):
    # Insert filler inside the user message, retaining the real chat wrapper.
    suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Briefly explain {tag} in English."}], tokenize=True, add_generation_prompt=True
    )
    if hasattr(suffix, "keys"):
        suffix = suffix["input_ids"]
    filler = tokenizer.encode(" context", add_special_tokens=False)[0]
    ids = suffix[:4] + [filler] * (length - len(suffix)) + suffix[4:]
    assert len(ids) == length
    begin = time.time()
    arrivals = []
    text = ""
    token_ids = []
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": model,
            "prompt": ids,
            "max_tokens": 160 if anchor else 64,
            "ignore_eos": True,
            "temperature": 0.7 if sampled else 0.0,
            "top_k": 16 if sampled else 1,
            "presence_penalty": 0.2 if sampled else 0.0,
            "stream": True,
            "return_token_ids": True,
        },
        stream=True,
        timeout=240,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        body = json.loads(line[6:])
        choices = body.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        chunk = choice.get("text", "")
        ids_out = choice.get("token_ids") or []
        if chunk or ids_out:
            arrivals.append(time.time())
            text += chunk
            token_ids += ids_out
            if anchor:
                first_token.set()
    end = time.time()
    return dict(
        length=length,
        tag=tag,
        sampled=sampled,
        prompt_token_ids=ids,
        start=begin,
        end=end,
        ttft_ms=1000 * (arrivals[0] - begin),
        max_decode_pause_ms=1000 * max(np.diff(arrivals), default=0),
        arrival_times=arrivals,
        text=text,
        token_ids=token_ids,
    )


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    anchor = pool.submit(run, 129, "ocean waves", False, True)
    if not first_token.wait(120):
        raise RuntimeError("Anchor produced no first token")
    pending = [
        pool.submit(run, n, tag, s)
        for n, tag, s in [
            (130, "volcanoes", False),
            (131, "rainbows", False),
            (513, "photosynthesis", False),
            (769, "gravity", True),
            (257, "magnets", False),
        ]
    ]
    results = [anchor.result()] + [f.result() for f in pending]
    pending = [
        pool.submit(run, n, tag, s)
        for n, tag, s in [
            (131, "rainbows", False),
            (129, "ocean waves", False),
            (130, "volcanoes", False),
            (513, "photosynthesis", False),
            (769, "gravity", True),
        ]
    ]
    results += [f.result() for f in pending]
finished = time.time()
all_events = events()
during = all_events[len(baseline) :]
captures = [e for e in all_events if e["event"] == "capture"]
counts = Counter((e["kind"], json.dumps(e["key"])) for e in captures)
retired = [e for e in during if "release" in e["event"] or "retire" in e["event"]]
chunks = [e for e in during if e["event"] == "prefill_chunk"]
decodes = [e for e in during if e["event"] == "decode_submit"]
before = next((e["counters"] for e in reversed(baseline) if "counters" in e), {})
after = next(e["counters"] for e in reversed(during) if "counters" in e)
checks = dict(
    one_capture_per_signature=all(n == 1 for n in counts.values()),
    zero_retirements=not retired,
    nonaligned_lengths={129, 130, 131}.issubset({r["length"] for r in results}),
    changed_continuation_starts=len({e["start"] for e in chunks if e["start"] > 0}) >= 2,
    decode_interleaved=bool(decodes) and any(decodes[0]["time"] < e["time"] < decodes[-1]["time"] for e in chunks),
    resident_decode=any(not e["reload_inputs"] for e in decodes),
    allocator_page_growth=any(e["reload_page_table"] for e in decodes),
    zero_logit_readbacks=after.get("logit_readbacks", 0) == before.get("logit_readbacks", 0),
)
# Compare the identical greedy prefixes across altered batch composition.
bytag = {}
for r in results:
    if not r["sampled"]:
        bytag.setdefault(r["tag"], []).append(r)
checks["greedy_repeat_tokens"] = all(
    bool(rows[0]["token_ids"]) and rows[0]["token_ids"][:64] == rows[1]["token_ids"][:64]
    for rows in bytag.values()
    if len(rows) > 1
)
summary = dict(
    status="pass" if all(checks.values()) else "fail",
    checks=checks,
    baseline_events=baseline,
    trace_captures=captures,
    capture_counts={str(k): v for k, v in counts.items()},
    workload={"requests": len(results), "max_concurrency": 6, "lengths": [r["length"] for r in results]},
    ttft_ms={str(p): float(np.percentile([r["ttft_ms"] for r in results], p)) for p in (50, 95, 99)},
    max_decode_pause_ms=max(r["max_decode_pause_ms"] for r in results),
    completed_requests_per_second=len(results) / (finished - started),
    results=results,
    event_range=[len(baseline), len(all_events)],
    events_path=str(events_path),
)
Path(a.output).write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {k: v for k, v in summary.items() if k not in ("results", "baseline_events", "trace_captures")}, indent=2
    )
)
if not all(checks.values()):
    raise SystemExit(1)
