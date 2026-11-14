import asyncio
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(__file__))
import gpqa_eval as g
from aiohttp import web

cases = [
    ("Answer: C", "C"),
    ("**Answer:** B", "B"),
    ("Answer: (D)", "D"),
    ("answer: a", "A"),
    ("ANSWER : $b", "B"),
    ("Answer: A\nbut actually\nAnswer: C", "C"),
    ("no final here", None),
    ("Answer: Z", None),
    ("Answer:D.", "D"),
    ("The answer is C", None),
    ("Answer: $(B)", "B"),
    ("", None),
    ("Answer:\nC", "C"),
]
for s, e in cases:
    assert g.extract_answer(s) == e, (s, g.extract_answer(s), e)
print("regex tests passed:", len(cases))

q1, q2 = g.load_questions(g.DEFAULT_CSV), g.load_questions(g.DEFAULT_CSV)
assert [q["correct_letter"] for q in q1] == [q["correct_letter"] for q in q2] and [q["prompt"] for q in q1] == [
    q["prompt"] for q in q2
]
print(
    "shuffle deterministic; n =",
    len(q1),
    "dist:",
    dict(sorted(collections.Counter(q["correct_letter"] for q in q1).items())),
)

idx = {q["record_id"]: i for i, q in enumerate(q1)}
prompt2id = {q["prompt"]: q["record_id"] for q in q1}
calls = [0]


async def models(r):
    return web.json_response({"data": [{"id": "fake"}]})


async def chat(r):
    body = await r.json()
    calls[0] += 1
    rid = prompt2id[body["messages"][0]["content"]]
    q = q1[idx[rid]]
    txt = f"reasoning...\nAnswer: {q['correct_letter']}" if idx[rid] % 2 == 0 else "Answer: Z"
    return web.json_response(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": txt}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10 + idx[rid], "total_tokens": 15},
        }
    )


def serve():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.add_routes([web.get("/v1/models", models), web.post("/v1/chat/completions", chat)])
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", 18765).start())
    loop.run_forever()


threading.Thread(target=serve, daemon=True).start()
import time

time.sleep(1)
d = tempfile.mkdtemp()
here = os.path.dirname(os.path.abspath(__file__))


def run(out, extra=()):
    r = subprocess.run(
        [
            sys.executable,
            f"{here}/gpqa_eval.py",
            "--base-url",
            "http://127.0.0.1:18765/v1",
            "--out",
            out,
            "--limit",
            "20",
            "--concurrency",
            "4",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


a = f"{d}/a.jsonl"
run(a)
n1 = calls[0]
s = json.load(open(a + ".summary.json"))
print("summary:", s)
assert s["accuracy"] == 0.5 and s["n"] == 20 and n1 == 20 and s["n_no_answer"] == 10
out2 = run(a)
assert calls[0] == n1, "resume made requests"
print("resume: 0 new requests; ", out2.splitlines()[0])
b = f"{d}/b.jsonl"
run(b, ["--seed-offset", "1"])
with open(b) as f:
    L = f.read().splitlines()
open(b, "w").write("\n".join(L[:15]) + "\n")
print(subprocess.run([sys.executable, f"{here}/compare.py", a, b], capture_output=True, text=True).stdout)
