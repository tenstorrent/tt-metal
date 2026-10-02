# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of a running server (stdlib only), mimicking what an agent harness sends:

    python models/demos/mimo_v2_d_p/serve/client_check.py [--url http://localhost:8000/v1] [--thinking 0|1]

1 GET /v1/models; 2 a plain completion; 3 the same streamed (SSE); 4 a tool-call round trip with a read_file tool
(the conversation resent in full on turn 2, as harnesses do, so the server's prefix cache is exercised); prints
TTFT / tok/s / cached tokens per request from the server's ``timings``.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the repository and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repository root"},
                "max_lines": {"type": "integer", "description": "Return at most this many lines"},
            },
            "required": ["path"],
        },
    },
}


def post(url, body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=3600)
    if not stream:
        return json.loads(r.read()), time.perf_counter() - t0, None
    first, chunks = None, []
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        c = json.loads(line[6:])
        d = (c.get("choices") or [{}])[0].get("delta", {}) if c.get("choices") else {}
        if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
            first = time.perf_counter() - t0
        chunks.append(c)
    return chunks, time.perf_counter() - t0, first


def show(tag, timings, wall):
    t = timings or {}
    print(
        f"[{tag}] wall {wall:.1f}s | prompt {t.get('prompt_tokens')} cached {t.get('cached_tokens')} "
        f"chunks {t.get('prefill_chunks')} | TTFT {t.get('ttft_s')}s | {t.get('completion_tokens')} tokens "
        f"@ {t.get('decode_tok_s')} tok/s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    kw = {"chat_template_kwargs": {"enable_thinking": bool(a.thinking)}, "max_tokens": a.max_tokens, "temperature": 0}

    print("models:", json.loads(urllib.request.urlopen(url + "/models").read()))

    msgs = [{"role": "user", "content": "In one sentence: what is a Tensix core?"}]
    r, wall, _ = post(url + "/chat/completions", {"model": "mimo", "messages": msgs, **kw})
    show("plain", r.get("timings"), wall)
    print("  ->", json.dumps(r["choices"][0], ensure_ascii=False)[:600])

    chunks, wall, ttft = post(
        url + "/chat/completions", {"model": "mimo", "messages": msgs, "stream": True, **kw}, True
    )
    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
    reas = "".join((c["choices"][0]["delta"].get("reasoning_content") or "") for c in chunks if c.get("choices"))
    show("stream", chunks[-1].get("timings"), wall)
    print(f"  client TTFT {ttft:.2f}s, {len(chunks)} SSE chunks, finish {chunks[-1]['choices'][0]['finish_reason']}")
    print("  reasoning:", reas[:200].replace("\n", " "), "| content:", text[:400])

    root = Path(__file__).parents[4]
    msgs = [
        {
            "role": "system",
            "content": "You are a coding agent working in the tt-metal repository. Use tools to inspect files.",
        },
        {
            "role": "user",
            "content": "Use the read_file tool to read models/demos/mimo_v2_d_p/README.md (max 40 lines), "
            "then tell me in two sentences what this model directory is.",
        },
    ]
    for turn in range(1, 4):
        r, wall, _ = post(url + "/chat/completions", {"model": "mimo", "messages": msgs, "tools": [READ_FILE], **kw})
        ch = r["choices"][0]
        show(f"tools turn {turn}", r.get("timings"), wall)
        print(f"  finish {ch['finish_reason']}:", json.dumps(ch["message"], ensure_ascii=False)[:600])
        msgs.append(ch["message"])
        if ch["finish_reason"] != "tool_calls":
            break
        for tc in ch["message"]["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            p = root / args.get("path", "")
            try:
                lines = p.read_text().splitlines()[: int(args.get("max_lines") or 40)]
                result = "\n".join(lines)
            except OSError as e:
                result = f"error: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # prefix reuse: the same long conversation once more with a new final question (cache hit up to the last turn)
    msgs.append({"role": "user", "content": "Now in one short sentence: which hardware does it target?"})
    r, wall, _ = post(url + "/chat/completions", {"model": "mimo", "messages": msgs, "tools": [READ_FILE], **kw})
    show("follow-up (prefix reuse)", r.get("timings"), wall)
    print("  ->", (r["choices"][0]["message"].get("content") or "")[:300])
    r, wall, _ = post(
        url + "/chat/completions",
        {"model": "mimo", "messages": msgs, "tools": [READ_FILE], "mimo_reset_cache": True, **kw},
    )
    show("follow-up (cache dropped)", r.get("timings"), wall)


if __name__ == "__main__":
    main()
