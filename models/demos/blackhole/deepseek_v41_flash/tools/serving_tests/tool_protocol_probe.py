#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tool-protocol probe: does the server's chat renderer carry tools, assistant tool_calls, tool results and thinking?

    python3 tool_protocol_probe.py [--base-url URL] [--model NAME] [--out probe.json]

Standard library only. Works with both renderers (the stock chat_template.jinja launch and tokenizer-mode deepseek_v41):
the prompt is inspected through the server's own `usage.prompt_tokens` (same message with and without the item under test,
max_tokens=1), never through /tokenize (optional, reported as a separate non-fatal line), and the end-to-end checks use a
random marker that only the tool result contains. Exit code 0 = every case PASS, 1 = at least one FAIL.

Cases: 1 schema_in_prompt, 2 tools_request_parsed, 3 assistant_tool_calls_in_prompt, 4 tool_result_in_prompt,
5 plain_chat_placement, 6 thinking_true_placement, 7 tool_loop_marker.
"""
import argparse
import json
import os
import random
import string
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather report for a city. The report contains a station code.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]
USER = {
    "role": "user",
    "content": "What is the weather in Paris right now? Call the get_weather tool, then tell me the station code.",
}
CALL = {
    "id": "call_0001",
    "type": "function",
    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
}
ASSISTANT_CALL = {"role": "assistant", "content": None, "tool_calls": [CALL]}


class Probe:
    def __init__(self, base, model):
        self.base, self.model, self.calls = base.rstrip("/"), model, []

    def post(self, path, body, timeout=900):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                st, data = r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            t = e.read().decode(errors="replace")
            try:
                st, data = e.code, json.loads(t)
            except ValueError:
                st, data = e.code, t
        except Exception as e:  # connection error
            st, data = -1, repr(e)
        self.calls.append({"path": path, "request": body, "status": st, "response": data})
        return st, data

    def chat(self, messages, tools=None, max_tokens=400, **extra):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(extra)
        return self.post("/v1/chat/completions", body)

    def prompt_tokens(self, messages, tools=None):
        st, r = self.chat(messages, tools, max_tokens=1)
        if st == 200 and isinstance(r, dict):
            return r["usage"]["prompt_tokens"], None
        return None, "HTTP %s %s" % (st, json.dumps(r)[:200])


def message(r):
    try:
        return r["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=None, help="served model name (default: first entry of /v1/models)")
    ap.add_argument("--out", default=None, help="write the full report (cases, raw requests and responses) as JSON")
    a = ap.parse_args()
    if a.model is None:
        try:
            with urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models", timeout=60) as r:
                a.model = json.loads(r.read())["data"][0]["id"]
        except Exception as e:
            print("cannot reach %s: %r" % (a.base_url, e))
            return 2
    p = Probe(a.base_url, a.model)
    marker = (
        "ZX-"
        + "".join(random.choice(string.digits) for _ in range(4))
        + "-"
        + "".join(random.choice(string.ascii_uppercase) for _ in range(2))
    )
    result_text = json.dumps({"temp_c": 17, "sky": "overcast", "station_code": marker})
    tool_msg = {"role": "tool", "tool_call_id": CALL["id"], "content": result_text}
    cases = []

    def case(name, ok, detail, guards):
        cases.append({"case": name, "result": "PASS" if ok else "FAIL", "detail": detail, "guards": guards})

    # 1 schema in prompt: tools add tokens to the prompt (the schema is rendered), same message without tools as baseline
    base_n, e0 = p.prompt_tokens([USER])
    tools_n, e1 = p.prompt_tokens([USER], TOOLS)
    ok = base_n is not None and tools_n is not None and tools_n - base_n >= 60
    case(
        "1_schema_in_prompt",
        ok,
        "prompt_tokens plain=%s with_tools=%s%s" % (base_n, tools_n, "" if ok else " err=%s" % (e1 or e0)),
        "declared tools reach the prompt (stock jinja template ignores them / server refuses tool_choice auto)",
    )

    # 2 tools request parsed into tool_calls
    st, r = p.chat([USER], TOOLS, max_tokens=500)
    m = message(r)
    tc = m.get("tool_calls") or []
    good = False
    if tc:
        try:
            good = tc[0]["function"]["name"] == "get_weather" and "city" in json.loads(tc[0]["function"]["arguments"])
        except (KeyError, ValueError, TypeError):
            good = False
    case(
        "2_tools_request_parsed",
        good,
        "HTTP %s tool_calls=%s content=%r" % (st, json.dumps(tc)[:160], (m.get("content") or "")[:80]),
        "tool-call-parser + enable-auto-tool-choice set and the model's DSML call is parsed into tool_calls",
    )

    # 3 assistant tool_calls (content null) in history are encoded: prompt grows vs the same history with an empty assistant turn
    follow = {"role": "user", "content": "Thanks."}
    n_with, e2 = p.prompt_tokens([USER, ASSISTANT_CALL, follow])
    n_wo, e3 = p.prompt_tokens([USER, {"role": "assistant", "content": ""}, follow])
    ok = n_with is not None and n_wo is not None and n_with - n_wo >= 8
    case(
        "3_assistant_tool_calls_in_prompt",
        ok,
        "prompt_tokens with_tool_calls=%s empty_assistant=%s%s" % (n_with, n_wo, "" if ok else " err=%s" % (e2 or e3)),
        "assistant tool_calls with content null are encoded (stock template drops them)",
    )

    # 4 tool result (OpenAI role:tool) is encoded: prompt grows with the result length
    long_result = {"role": "tool", "tool_call_id": CALL["id"], "content": result_text + " " + " ".join(["detail"] * 40)}
    n_res, e4 = p.prompt_tokens([USER, ASSISTANT_CALL, long_result])
    n_nores, e5 = p.prompt_tokens([USER, ASSISTANT_CALL, {"role": "tool", "tool_call_id": CALL["id"], "content": "x"}])
    ok = n_res is not None and n_nores is not None and n_res - n_nores >= 40
    case(
        "4_tool_result_in_prompt",
        ok,
        "prompt_tokens long_result=%s short_result=%s%s" % (n_res, n_nores, "" if ok else " err=%s" % (e4 or e5)),
        "role:tool messages with tool_call_id are encoded (stock template drops them)",
    )

    # 5 plain chat, no thinking kwarg: answer in content, reasoning empty
    q = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
    st, r = p.chat(q, max_tokens=300)
    m = message(r)
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    ok = st == 200 and "391" in (m.get("content") or "") and not reasoning
    case(
        "5_plain_chat_placement",
        ok,
        "HTTP %s content=%r reasoning=%r" % (st, (m.get("content") or "")[:60], (reasoning or "")[:60]),
        "non-thinking default (default-chat-template-kwargs thinking=false): the answer lands in content, not in reasoning",
    )

    # 6 thinking=true: reasoning field filled, answer in content
    st, r = p.chat(q, max_tokens=3000, chat_template_kwargs={"thinking": True})
    m = message(r)
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    ok = st == 200 and bool(reasoning) and "391" in (m.get("content") or "")
    case(
        "6_thinking_true_placement",
        ok,
        "HTTP %s reasoning=%r content=%r" % (st, (reasoning or "")[:60], (m.get("content") or "")[:60]),
        "thinking=true reaches the renderer and reasoning-parser splits reasoning from content",
    )

    # 7 full tool loop; the marker exists only in the tool result
    st, r = p.chat([USER], TOOLS, max_tokens=500)
    m1 = message(r)
    tcs = m1.get("tool_calls") or []
    final, ok, detail = "", False, "turn 1 produced no tool_calls (HTTP %s)" % st
    if tcs:
        msgs = [USER, {"role": "assistant", "content": m1.get("content"), "tool_calls": tcs}]
        msgs += [{"role": "tool", "tool_call_id": c["id"], "content": result_text} for c in tcs]
        st2, r2 = p.chat(msgs, TOOLS, max_tokens=500)
        m2 = message(r2)
        final = m2.get("content") or ""
        ok = st2 == 200 and marker in final and not m2.get("tool_calls")
        detail = "HTTP %s marker=%s final=%r" % (st2, marker, final[:120])
    case(
        "7_tool_loop_marker",
        ok,
        detail,
        "end to end: call, history, tool result are all encoded; the model's final answer quotes the marker only the tool result contains",
    )

    # optional /tokenize (not all renderers support history messages there): separate, never fails the run
    tok = {}
    st, r = p.post(
        "/tokenize",
        {"model": a.model, "messages": [USER, ASSISTANT_CALL, tool_msg], "tools": TOOLS, "add_generation_prompt": True},
    )
    tok["status"] = st
    if st == 200 and isinstance(r, dict) and "tokens" in r:
        st2, d = p.post("/detokenize", {"model": a.model, "tokens": r["tokens"]})
        prompt = d.get("prompt", "") if isinstance(d, dict) else ""
        tok["marker_in_prompt"] = marker in prompt
        tok["rendered_prompt"] = prompt
    else:
        tok["error"] = json.dumps(r)[:200]

    for c in cases:
        print("%-4s %-34s %s" % (c["result"], c["case"], c["detail"]))
    print(
        "INFO /tokenize (optional, non-fatal): %s"
        % json.dumps({k: v for k, v in tok.items() if k != "rendered_prompt"})
    )
    nfail = sum(c["result"] == "FAIL" for c in cases)
    print("RESULT %s (%d/%d cases pass)" % ("PASS" if nfail == 0 else "FAIL", len(cases) - nfail, len(cases)))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {
                    "base_url": a.base_url,
                    "model": a.model,
                    "marker": marker,
                    "cases": cases,
                    "tokenize_optional": tok,
                    "raw": p.calls,
                },
                f,
                indent=1,
                ensure_ascii=False,
            )
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
