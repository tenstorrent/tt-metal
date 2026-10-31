#!/usr/bin/env python3
"""Streaming structured outputs / tool calls: incremental deltas, finish reasons, usage (vs the same request non-streamed, greedy).

usage: stream_check.py [--host H] [--port P] [--rounds R] [--par P] [--out FILE]
Per case: one non-stream request and one stream request (stream_options.include_usage) of the same greedy body; checks
  - first chunk carries role=assistant, exactly one finish_reason chunk, it is the last choice chunk, usage chunk after it has no choices, [DONE] last
  - concatenated content / tool arguments equal the non-stream result (WARN, not FAIL, when both are grammar-valid and differ: bf16 near-ties)
  - content / arguments valid against the grammar, finish_reason equal to the non-stream one (tool cases: tool_calls, structured: stop, cut: length)
  - usage: prompt_tokens equal, completion_tokens equal to the non-stream value, total = prompt + completion
  - tool deltas: id and name only in the first delta of an index, indexes 0.., no content leaked next to tool calls
  - more than one content / arguments delta (incremental) when the text is longer than 24 characters
--par P also runs every case P times concurrently (cross-talk check).
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrix as M  # noqa: E402


def post(a, body, stream):
    body = dict(body)
    body.update(body.pop("extra_body", {}))
    body.update(model=M.MODEL, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (a.host, a.port),
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return {"http": e.code, "error": e.read().decode()[:300]}
    if not stream:
        d = json.loads(r.read())
        m = d["choices"][0]["message"]
        return {
            "http": 200,
            "content": m.get("content") or "",
            "calls": m.get("tool_calls") or [],
            "finish": d["choices"][0]["finish_reason"],
            "usage": d.get("usage"),
        }
    ev, done, t0 = [], False, time.time()
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            done = True
            continue
        if done:
            ev.append(("after_done", None, time.time() - t0))
        ev.append(("chunk", json.loads(line[5:]), time.time() - t0))
    return {"http": 200, "events": ev, "done": done}


def analyse(res, nonstream, check, want_finish, tool):
    fails, warns = [], []
    ev = [e for e in res["events"]]
    if any(k == "after_done" for k, _, _ in ev):
        fails.append("data after [DONE]")
    chunks = [(d, t) for k, d, t in ev if k == "chunk"]
    if not res["done"]:
        fails.append("no [DONE]")
    for d, _ in chunks:
        if "error" in d:
            return ["SSE error event %s" % json.dumps(d["error"])[:200]], warns, {}
    ch = [(d, t) for d, t in chunks if d.get("choices")]
    if not ch:
        return ["no choice chunks"], warns, {}
    if (ch[0][0]["choices"][0].get("delta") or {}).get("role") != "assistant":
        fails.append("first chunk has no role=assistant")
    fin = [i for i, (d, _) in enumerate(ch) if d["choices"][0].get("finish_reason")]
    if len(fin) != 1:
        fails.append("%d finish_reason chunks" % len(fin))
    elif fin[0] != len(ch) - 1:
        fails.append("finish_reason chunk is not the last choice chunk (%d of %d)" % (fin[0], len(ch)))
    finish = ch[fin[-1]][0]["choices"][0]["finish_reason"] if fin else None
    if finish != want_finish:
        fails.append("finish_reason %r, wanted %r" % (finish, want_finish))
    if nonstream.get("finish") != finish:
        fails.append("finish_reason stream %r != non-stream %r" % (finish, nonstream.get("finish")))
    uchunks = [d for d, _ in chunks if d.get("usage")]
    if len(uchunks) != 1:
        fails.append("%d usage chunks" % len(uchunks))
    else:
        u = uchunks[0]
        if u.get("choices"):
            fails.append("usage chunk carries choices")
        if chunks[-1][0] is not u:
            fails.append("usage chunk is not the last chunk")
        u = u["usage"]
        n = nonstream.get("usage") or {}
        if u.get("total_tokens") != u.get("prompt_tokens", 0) + u.get("completion_tokens", 0):
            fails.append("usage total != prompt + completion: %s" % u)
        if u.get("prompt_tokens") != n.get("prompt_tokens"):
            fails.append("prompt_tokens %s != %s" % (u.get("prompt_tokens"), n.get("prompt_tokens")))
        if u.get("completion_tokens") != n.get("completion_tokens"):
            (fails if nonstream["content"] == res["content"] and nonstream["calls"] == res["calls"] else warns).append(
                "completion_tokens stream %s != non-stream %s"
                % (u.get("completion_tokens"), n.get("completion_tokens"))
            )
    return fails, warns, {"chunks": len(ch), "finish": finish}


def collect(res):
    content, parts, calls, ndelta, ntool, leak = [], [], {}, 0, 0, False
    prev_idx = -1
    for k, d, t in res["events"]:
        if k != "chunk" or not d.get("choices"):
            continue
        dl = d["choices"][0].get("delta") or {}
        if dl.get("content"):
            content.append(dl["content"])
            ndelta += 1
        for tc in dl.get("tool_calls") or []:
            i = tc["index"]
            c = calls.setdefault(i, {"id": None, "name": "", "args": [], "first": True, "bad": []})
            fn = tc.get("function") or {}
            if not c["first"]:
                if tc.get("id"):
                    c["bad"].append("id repeated in a later delta")
                if fn.get("name"):
                    c["bad"].append("name repeated in a later delta")
            if tc.get("id"):
                c["id"] = tc["id"]
            if fn.get("name"):
                c["name"] += fn["name"]
            if fn.get("arguments"):
                c["args"].append(fn["arguments"])
                ntool += 1
            c["first"] = False
            if i < prev_idx:
                c["bad"].append("index went backwards")
            prev_idx = max(prev_idx, i)
    res["content"] = "".join(content)
    res["calls"] = [
        {
            "id": calls[i]["id"],
            "type": "function",
            "function": {"name": calls[i]["name"], "arguments": "".join(calls[i]["args"])},
        }
        for i in sorted(calls)
    ]
    res["bad"] = [b for i in calls for b in calls[i]["bad"]] + (
        ["tool call without id"] if any(not calls[i]["id"] for i in calls) else []
    )
    res["ndelta"] = ndelta
    res["ntool"] = ntool
    return res


CUT = {"max_tokens": 8}
STREAM_CASES = [  # (name, body key in matrix.CASES, extra body, wanted finish, is_tool)
    ("json_schema", "json_schema", {}, "stop", False),
    ("json_object", "json_object", {}, "stop", False),
    ("regex", "regex", {}, "stop", False),
    ("choice", "choice", {}, "stop", False),
    ("required", "required", {}, "tool_calls", True),
    (
        "named",
        "named",
        {},
        "stop",
        True,
    ),  # vLLM reports finish_reason "stop" for a named tool_choice (same in m0/m1/m2, stream and non-stream)
    ("json_schema_cut8", "json_schema", CUT, "length", False),
    ("required_cut8", "required", CUT, "length", True),
]


def run_case(a, case):
    name, key, extra, want, tool = case
    body, check = M.CASES[key]
    body = dict(body, temperature=0.0, max_tokens=extra.get("max_tokens", 384))
    ns = post(a, body, False)
    if ns.get("http") != 200:
        return name, "FAIL", ["non-stream HTTP %s %s" % (ns.get("http"), ns.get("error"))], [], {}
    s = post(a, body, True)
    if s.get("http") != 200:
        return name, "FAIL", ["stream HTTP %s %s" % (s.get("http"), s.get("error"))], [], {}
    if any(k == "chunk" and "error" in d for k, d, _ in s["events"]):
        return name, "FAIL", ["SSE error event"], [], {}
    collect(s)
    fails, warns, info = analyse(s, ns, check, want, tool)
    fails += s["bad"]
    if want == "length":
        pass  # a cut grammar prefix is not valid output; only the protocol is checked
    else:
        msg = {"content": s["content"], "tool_calls": s["calls"]}
        why = check(msg) if check else None
        if why:
            fails.append("stream output invalid: %s" % why)
        if tool and s["content"].strip():
            fails.append("content next to tool calls: %r" % s["content"][:80])
        if s["content"] != ns["content"] or [c["function"] for c in s["calls"]] != [c["function"] for c in ns["calls"]]:
            warns.append(
                "stream text differs from non-stream (both valid): %r vs %r"
                % (
                    (s["content"] or json.dumps([c["function"] for c in s["calls"]]))[:90],
                    (ns["content"] or json.dumps([c["function"] for c in ns["calls"]]))[:90],
                )
            )
    text = s["content"] if not tool else "".join(c["function"]["arguments"] for c in s["calls"])
    nd = s["ntool"] if tool else s["ndelta"]
    if len(text) > 24 and nd < 2 and want != "length":
        fails.append("not incremental: %d deltas for %d chars" % (nd, len(text)))
    info.update(deltas=nd, chars=len(text))
    return name, "FAIL" if fails else "PASS", fails, warns, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = [c for _ in range(a.rounds) for c in STREAM_CASES for _ in range(a.par)]
    with cf.ThreadPoolExecutor(max(1, a.par * 2)) as ex:
        res = list(ex.map(lambda c: run_case(a, c), jobs))
    lines = []
    for name, st, fails, warns, info in res:
        lines.append(
            "%-4s %-18s %s %s %s"
            % (st, name, json.dumps(info), "; ".join(fails), ("WARN " + "; ".join(warns)) if warns else "")
        )
    bad = [r for r in res if r[1] != "PASS"]
    lines.append("SUMMARY %d/%d pass, %d with warnings" % (len(res) - len(bad), len(res), sum(1 for r in res if r[3])))
    out = "\n".join(lines)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
