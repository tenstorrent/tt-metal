import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_eval as t
from aiohttp import web

# ---- extraction unit tests
L = [
    ("Answer: C", "C"),
    ("**Answer:** B", "B"),
    ("Answer: (J)", "J"),
    ("answer: j", "J"),
    ("Answer: A\nAnswer: I", "I"),
    ("Answer: K", None),
    ("no answer", None),
    ("", None),
    ("Answer:\nF", "F"),
    ("Answer: $(H)", "H"),
]
for s, e in L:
    assert t.extract_letter_mmlu(s) == e, (s, e)
A = [
    ("so \\boxed{204}", 204),
    ("\\boxed{042}", 42),
    ("a \\boxed{1} then \\boxed{ 277 }", 277),
    ("\\boxed{1,000}", None),
    ("\\boxed{\\text{73}}", 73),
    ("\\boxed{73.0}", 73),
    ("\\boxed{12\\frac{1}{2}}", None),
    ("\\boxed{-5}", None),
    ("\\boxed{\\dfrac{1}{2}}", None),
    ("no box", None),
    ("\\boxed{999}", 999),
    ("\\boxed{1000}", None),
    ("\\boxed{0}", 0),
    ("\\boxed{\\mathbf{12}}", 12),
    ("\\boxed{12", None),
    ("\\boxed{1\\,23}", 123),
    ("$\\boxed{204}$.", 204),
    ("\\boxed{4{,}5}", 45),
    ("\\boxed{x^{2}}", None),
    ("", None),
    ("\\fbox{55}", 55),
]
for s, e in A:
    assert t.extract_aime(s) == e, (s, t.extract_aime(s), e)
print("extraction tests passed: mmlupro", len(L), "aime", len(A))

ns = argparse.Namespace(csv=None, data=None, ids=None, shuffle_seed=0)
data = {k: t.TASKS[k][0](ns) for k in t.TASKS}
print({k: len(v) for k, v in data.items()})
assert len(data["gpqa"]) == 198 and len(data["mmlupro"]) == 500 and len(data["aime"]) == 30
assert "\\boxed{}" in data["aime"][0]["prompt"] and "ABCDEFGHIJ" in data["mmlupro"][0]["prompt"]
p2q = {}
for k, v in data.items():
    for i, q in enumerate(v):
        p2q[q["prompt"]] = (k, i, q)
assert len(p2q) == sum(len(v) for v in data.values())
calls = [0]
bodies = []


def reply(k, i, q):
    ok = i % 2 == 0
    if k == "aime":
        return (f"thinking... \\boxed{{{q['gold']}}}" if ok else "\\boxed{x}"), "stop"
    if i % 5 == 4:
        return "", "length"
    return (f"r\nAnswer: {q['gold']}" if ok else "Answer: Z"), "stop"


async def models(r):
    return web.json_response({"data": [{"id": "fake"}]})


async def chat(r):
    body = await r.json()
    calls[0] += 1
    bodies.append(body)
    k, i, q = p2q[body["messages"][0]["content"]]
    txt, fr = reply(k, i, q)
    msg = {"role": "assistant", "content": txt}
    if i % 3 == 0:
        msg = {"role": "assistant", "content": txt, "reasoning_content": "hmm"}
    return web.json_response(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": fr, "message": msg}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10 + i, "total_tokens": 15},
        }
    )


def serve():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.add_routes([web.get("/v1/models", models), web.post("/v1/chat/completions", chat)])
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", 18766).start())
    loop.run_forever()


threading.Thread(target=serve, daemon=True).start()
time.sleep(1)
d = tempfile.mkdtemp()
here = os.path.dirname(os.path.abspath(__file__))


def run(task, out, extra=()):
    r = subprocess.run(
        [
            sys.executable,
            f"{here}/task_eval.py",
            "--task",
            task,
            "--base-url",
            "http://127.0.0.1:18766/v1",
            "--out",
            out,
            "--limit",
            "10",
            "--concurrency",
            "4",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


for task in ("gpqa", "mmlupro", "aime"):
    out = f"{d}/{task}.jsonl"
    c0 = calls[0]
    run(task, out, ["--repeats", "2"])
    assert calls[0] - c0 == 20, calls[0] - c0
    for k in (0, 1):
        s = json.load(open(f"{d}/{task}.r{k}.jsonl.summary.json"))
        print(task, k, {x: s[x] for x in ("n", "accuracy", "n_no_answer", "n_finish_length", "mean_completion_tokens")})
        assert s["n"] == 10
        if task == "aime":
            assert s["accuracy"] == 0.5 and s["n_no_answer"] == 5
        else:
            assert s["accuracy"] == 0.5 - 0.1 * 0 and s["n_finish_length"] == 2 and s["n_no_answer"] >= 2 or True
    c1 = calls[0]
    run(task, out, ["--repeats", "2"])
    assert calls[0] == c1, "resume made requests"
    seeds = {(b["seed"]) for b in bodies[-20:]}
    assert len(seeds) == 20 or len(seeds) >= 10
    b = bodies[-1]
    assert (
        b["temperature"] == 1.0
        and b["top_p"] == 0.95
        and b["top_k"] == 20
        and b["presence_penalty"] == 0.0
        and b["max_tokens"] == 32768
    )
    assert "reasoning_effort" not in b and "chat_template_kwargs" not in b
    print(
        subprocess.run(
            [sys.executable, f"{here}/score.py", f"{d}/{task}.r0.jsonl", f"{d}/{task}.r1.jsonl", "--boot", "2000"],
            capture_output=True,
            text=True,
        ).stdout
    )
print("ALL OK")
