#!/usr/bin/env python3
"""Structured-output matrix against an OpenAI-compatible server: (case) x (stream, non-stream) x (greedy, sampled).

Every structured output must validate against its grammar / schema; tool calls must round-trip through the server's tool parser.
usage: matrix.py [--host H] [--port P] [--workers N] [--filler F] [--cases a,b] [--max-tokens M] [--out FILE]
  --workers N  run the cells N at a time (a mixed batch); --filler F also keeps F plain unstructured chat requests in flight.
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request

import jsonschema

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
WEATHER = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    "required": ["city", "unit"],
    "additionalProperties": False,
}
SEARCH = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
    "additionalProperties": False,
}


def tools(strict=False):
    f = lambda n, d, p: {
        "type": "function",
        "function": {"name": n, "description": d, "parameters": p, **({"strict": True} if strict else {})},
    }
    return [f("get_weather", "Current weather in a city", WEATHER), f("web_search", "Search the web", SEARCH)]


PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "tags"],
    "additionalProperties": False,
}
REGEX = r"[A-Z]{3}-[0-9]{4}"
CHOICES = ["alpha", "bravo", "charlie"]


def _tool_check(name=None):
    def check(msg):
        calls = msg.get("tool_calls") or []
        if not calls:
            return "no tool call parsed; content=%r" % (msg.get("content") or "")[:200]
        for c in calls:
            fn = c["function"]
            schema = {"get_weather": WEATHER, "web_search": SEARCH}.get(fn["name"])
            if schema is None:
                return "unknown tool %r" % fn["name"]
            if name and fn["name"] != name:
                return "wanted %s got %s" % (name, fn["name"])
            try:
                jsonschema.validate(json.loads(fn["arguments"]), schema)
            except Exception as e:
                return "arguments %r invalid: %s" % (fn["arguments"][:200], str(e)[:120])
        return None

    return check


def _json_check(schema):
    def check(msg):
        try:
            jsonschema.validate(json.loads(msg.get("content") or ""), schema)
        except Exception as e:
            return "content %r invalid: %s" % ((msg.get("content") or "")[:200], str(e)[:120])

    return check


def _obj_check(msg):
    try:
        v = json.loads(msg.get("content") or "")
        return None if isinstance(v, dict) else "not an object"
    except Exception as e:
        return "content %r not json: %s" % ((msg.get("content") or "")[:200], e)


def _regex_check(msg):
    c = (msg.get("content") or "").strip()
    return None if re.fullmatch(REGEX, c) else "content %r does not match %s" % (c[:100], REGEX)


def _choice_check(msg):
    c = (msg.get("content") or "").strip()
    return None if c in CHOICES else "content %r not in %s" % (c[:100], CHOICES)


WQ = [{"role": "user", "content": "What's the weather in Paris in celsius? Use the tool."}]
CASES = {
    "required": (dict(messages=WQ, tools=tools(), tool_choice="required"), _tool_check()),
    "named": (
        dict(messages=WQ, tools=tools(), tool_choice={"type": "function", "function": {"name": "web_search"}}),
        _tool_check("web_search"),
    ),
    "strict_auto": (
        dict(messages=WQ, tools=tools(strict=True), tool_choice="auto"),
        None,
    ),  # may answer in text: only a tool call, if made, is validated
    "json_schema": (
        dict(
            messages=[{"role": "user", "content": "Invent a person."}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": PERSON, "strict": True},
            },
        ),
        _json_check(PERSON),
    ),
    "json_object": (
        dict(
            messages=[{"role": "user", "content": "Reply with a JSON object describing a cat."}],
            response_format={"type": "json_object"},
        ),
        _obj_check,
    ),
    "regex": (
        dict(
            messages=[{"role": "user", "content": "Give me a ticket id."}],
            extra_body={"structured_outputs": {"regex": REGEX}},
        ),
        _regex_check,
    ),
    "choice": (
        dict(
            messages=[{"role": "user", "content": "Pick one."}], extra_body={"structured_outputs": {"choice": CHOICES}}
        ),
        _choice_check,
    ),
}


def post(host, port, body, stream, timeout=600):
    body = dict(body)
    extra = body.pop("extra_body", {})
    body.update(extra)
    body.update(model=MODEL, stream=stream)
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (host, port),
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"http": e.code, "error": e.read().decode()[:300]}
    if not stream:
        d = json.loads(r.read())
        return {
            "http": 200,
            "msg": d["choices"][0]["message"],
            "finish": d["choices"][0]["finish_reason"],
            "usage": d.get("usage"),
        }
    content, calls, finish, err, ttft, usage = [], {}, None, None, None, None
    t_req = time.time()
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if ttft is None and "error" not in d:
            ttft = time.time() - t_req
        if "error" in d:
            err = d["error"]
            continue
        if d.get("usage"):
            usage = d["usage"]
        if not d.get("choices"):
            continue
        ch = d["choices"][0]
        dl = ch.get("delta", {})
        if dl.get("content"):
            content.append(dl["content"])
        for tc in dl.get("tool_calls") or []:
            c = calls.setdefault(tc["index"], {"function": {"name": "", "arguments": ""}})
            c["function"]["name"] += tc.get("function", {}).get("name") or ""
            c["function"]["arguments"] += tc.get("function", {}).get("arguments") or ""
        finish = ch.get("finish_reason") or finish
    if err:
        return {"http": 200, "sse_error": err}
    return {
        "http": 200,
        "msg": {"content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)]},
        "finish": finish,
        "ttft": ttft,
        "usage": usage,
    }


def run_cell(a, name, stream, temp):
    body, check = CASES[name]
    body = dict(body, temperature=temp, max_tokens=a.max_tokens)
    if temp > 0:
        body["seed"] = 7
    t0 = time.time()
    res = post(a.host, a.port, body, stream)
    cell = "%s/%s/%s" % (name, "stream" if stream else "plain", "greedy" if temp == 0 else "sampled")
    if res.get("http") != 200:
        return cell, "FAIL", "HTTP %s %s" % (res["http"], res.get("error")), time.time() - t0
    if "sse_error" in res:
        return cell, "FAIL", "SSE error event %s" % json.dumps(res["sse_error"])[:200], time.time() - t0
    msg = res["msg"]
    if name == "strict_auto":
        why = _tool_check()(msg) if msg.get("tool_calls") else None
    else:
        why = check(msg)
    if res.get("finish") == "length" and why is None and name not in ("strict_auto",):
        why = None  # still valid against the grammar prefix rule; the schema check above already passed
    tt = res.get("ttft")
    return cell, ("PASS" if why is None else "FAIL"), (why or "") + (" ttft=%.1fs" % tt if tt else ""), time.time() - t0


def filler(a, stop):
    n = toks = 0
    while not stop.is_set():
        r = post(
            a.host,
            a.port,
            dict(
                messages=[{"role": "user", "content": "Write a short story about the sea."}],
                max_tokens=96,
                temperature=0,
                ignore_eos=True,
            ),
            False,
        )
        if r.get("http") == 200:
            n += 1
            toks += (r.get("usage") or {}).get("completion_tokens", 0)
    return (n, toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--filler", type=int, default=0)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    cells = [(n, s, t) for _ in range(a.rounds) for n in a.cases.split(",") for s in (False, True) for t in (0.0, 0.8)]
    import threading

    stop = threading.Event()
    t_start = time.time()
    with cf.ThreadPoolExecutor(a.workers + a.filler) as ex:
        fl = [ex.submit(filler, a, stop) for _ in range(a.filler)]
        futs = [ex.submit(run_cell, a, *c) for c in cells]
        results = [f.result() for f in futs]
        t_cells = time.time() - t_start
        stop.set()
        done = [f.result() for f in fl]
        t_all = time.time() - t_start
    bad = [r for r in results if r[1] != "PASS"]
    lines = ["%-4s %-28s %6.1fs %s" % (r[1], r[0], r[3], r[2]) for r in results]
    lines.append(
        "SUMMARY %d/%d pass; cells took %.0fs; filler (requests, tokens) %s over %.0fs: plain %.1f tok/s total"
        % (len(results) - len(bad), len(results), t_cells, done, t_all, sum(d[1] for d in done) / t_all if done else 0)
    )
    out = "\n".join(lines)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
