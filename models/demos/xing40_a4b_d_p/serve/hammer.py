# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness / stability hammer for the Xing serve stack (serve/README.md). Standard library only: runs anywhere
that reaches the server (on the box, or through an ssh tunnel).

    python models/demos/xing40_a4b_d_p/serve/hammer.py [--url http://localhost:8000] [--concurrency 2]
        [--rounds 1 | --duration-min 60] [--long 20000] [--max-tokens 16]

Cases, all greedy (temperature 0) so every answer is reproducible; the short ones start with a ~150-token system turn
so each step remounts the slot's resident prefix:
  fact      one-turn questions with one right answer (the answer must contain the expected word)
  memory    two turns: a number to remember, then a question about it (turn 2 remounts turn 1's slot)
  repeat    one prompt sent twice: the token ids must be identical (the second run starts from the reused prefix,
            so its chunk starts differ from the cold first run's)
  long      (--long N) a ~N-token haystack with one needle sentence, then a question about it: several chunks,
            the pulled-back last chunk when N is past 51200, then decode steps that remount mid-chunk
Workers run the cases concurrently (the engine interleaves their chunks over its slots). At the end: pass / fail
per case, step latency, and the server's /stats (its own per-step checks: check_fail must stay 0).

--suite creative adds cases that probe the stack from other angles; each failure is tagged "stack" (the serving path
is wrong: HTTP errors, nondeterminism, a limit not honoured) or "model" (an answer a 29B model can get wrong):
  chain       6 turns of personal facts, then one question recalling all of them (each turn remounts the last)
  retrieve    100-1500 "name: number" records scattered in filler (2K-40K tokens), three lookups in one question
  copy        repeat a list of 40 random 3-digit numbers exactly (a long decode across many 64-token blocks)
  boundary    a prompt that ends just before a 5120-token chunk edge, then a decode that crosses it
  math        (a + b) * c - d with small numbers                      [model]
  json        a JSON object with given fields, parsed                 [model]
  sort        eight words sorted alphabetically                       [model]
  unicode     repeat CJK / Greek / emoji text exactly                 [model]
  thinking    enable_thinking: reasoning_content and a right answer   [model]
  raw         /v1/completions continuation                            [model]
  seeded      temperature 0.9 + seed, twice: identical token ids      [stack]
  twins       the same prompt from two threads at once: identical ids [stack]
  stop        a stop string ends the answer before it, finish "stop"  [stack]
  cancel      a stream closed after 3 chunks; the next request works  [stack]
  oversize    a ~67K-token prompt (past max context): HTTP 400        [stack]
  maxctx      a prompt 8-40 tokens short of max context: finish "length" exactly at 56320, no error [stack]
Exit 1 if any stack failure, or the server counted a check failure / engine error; model failures are listed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

FACTS = [
    ("What is the capital of France? Answer in one word.", "Paris"),
    ("What is the capital of Japan? Answer in one word.", "Tokyo"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", "Au"),
    ("How many days are in a week? Answer with a number only.", "7"),
    ("What is 12 times 12? Answer with a number only.", "144"),
    ("Which planet is known as the Red Planet? Answer in one word.", "Mars"),
    ("What is the largest ocean on Earth? Answer in one or two words.", "Pacific"),
    ("Who wrote 'Romeo and Juliet'? Answer with the surname only.", "Shakespeare"),
]
# A neutral system turn (tests/bringup/contract/test_runner_smoke.py PAD_SYSTEM) in front of the short cases: past 64
# tokens, so every decode step remounts a resident prefix (block 64) instead of going cold (--no-system: off).
SYSTEM = (
    "You are a helpful, knowledgeable assistant. You answer questions about geography, history, science and everyday "
    "life. Read each question carefully before you answer. Keep your answers short and accurate, and follow any format "
    "the user asks for, such as a single word, a number or a short list. If a question has one clear answer, give that "
    "answer directly without extra explanation. If the user asks for one word, reply with exactly one word. Do not "
    "repeat the question, do not add greetings, and do not describe your reasoning. Use plain language and common "
    "spellings of names and places."
)
FILLER = (
    "The committee met on a grey morning to review the quarterly figures for the regional offices. Each office "
    "reported its staffing, its travel costs and the state of its equipment, and the minutes record every figure in "
    "the order in which it was read out. Nothing unusual was noted, and the meeting closed on time. "
)


SYSTEM_ON = True


def post(url, body, timeout):
    hdr = {"Content-Type": "application/json", "X-Xing-Pool": "hammer"}  # server.py POOLS: never the web chat's share
    req = urllib.request.Request(url, json.dumps(body).encode(), hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, messages, max_tokens, timeout, system=True, **extra):
    """-> (content, token ids, timings, seconds); extra: request fields (temperature, seed, stop, ...)."""
    t0 = time.time()
    if system and SYSTEM_ON:
        messages = [{"role": "system", "content": SYSTEM}] + messages
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra}
    r = post(f"{base}/v1/chat/completions", body, timeout)
    ch = r["choices"][0]
    tm = r.get("timings", {})
    tm["finish"], tm["reasoning"] = ch.get("finish_reason"), ch["message"].get("reasoning_content") or ""
    tm["usage"] = r.get("usage", {})
    return ch["message"].get("content") or "", ch.get("token_ids", []), tm, time.time() - t0


class Results:
    def __init__(self):
        self.lock = threading.Lock()
        self.by_case = {}
        self.fails = []  # stack failures (and every failure of the default suite)
        self.model_fails = []

    def add(self, case, ok, detail, kind="stack"):
        with self.lock:
            p, f = self.by_case.get(case, (0, 0))
            self.by_case[case] = (p + ok, f + (not ok))
            if not ok:
                (self.model_fails if kind == "model" else self.fails).append(f"{case}: {detail}")
        tag = "ok  " if ok else ("MISS" if kind == "model" else "FAIL")
        print(f"[hammer] {time.strftime('%H:%M:%S')} {tag} {case}: {detail}", flush=True)


def case_fact(a, rng, res):
    q, want = rng.choice(FACTS)
    text, _, tm, dt = chat(a.url, [{"role": "user", "content": q}], a.max_tokens, a.timeout)
    res.add("fact", want.lower() in text.lower(), f"{q!r} -> {text!r} ({dt:.0f} s, {tm.get('steps')} steps)")


def case_memory(a, rng, res):
    n = rng.randint(1000, 9999)
    msgs = [{"role": "user", "content": f"Remember this number: {n}. Reply only with 'Noted.'"}]
    t1, _, _, _ = chat(a.url, msgs, a.max_tokens, a.timeout)
    msgs += [
        {"role": "assistant", "content": t1},
        {"role": "user", "content": "Which number did I ask you to " "remember? Answer with the number only."},
    ]
    t2, _, tm, dt = chat(a.url, msgs, a.max_tokens, a.timeout)
    first = tm.get("first_step") or {}
    res.add("memory", str(n) in t2, f"{n} -> {t1!r} / {t2!r} ({dt:.0f} s, turn-2 resident {first.get('resident')})")


def case_repeat(a, rng, res):
    q, _ = rng.choice(FACTS)
    q = f"{q} Then explain your answer in one sentence."
    m = [{"role": "user", "content": q}]
    t1, ids1, tm1, _ = chat(a.url, m, a.max_tokens, a.timeout)
    t2, ids2, tm2, _ = chat(a.url, m, a.max_tokens, a.timeout)
    r1, r2 = (tm1.get("first_step") or {}).get("resident"), (tm2.get("first_step") or {}).get("resident")
    if ids1 == ids2:
        res.add("repeat", True, f"{len(ids1)} identical tokens (first-step resident {r1} then {r2})")
    else:
        k = next((i for i, (x, y) in enumerate(zip(ids1, ids2)) if x != y), min(len(ids1), len(ids2)))
        res.add("repeat", False, f"diverged at token {k} (resident {r1} vs {r2}): {t1!r} vs {t2!r}")


def case_long(a, rng, res):
    n_fill = max(1, a.long // 61)  # ~61 tokens per FILLER (22184 tokens for 363)
    code = rng.randint(100000, 999999)
    at = rng.randint(0, n_fill - 1)
    parts = [FILLER] * n_fill
    parts[at] = f"The secret access code for the archive room is {code}. " + FILLER
    msgs = [
        {
            "role": "user",
            "content": "".join(parts) + "\n\nWhat is the secret access code for the archive room? "
            "Answer with the number only.",
        }
    ]
    text, _, tm, dt = chat(a.url, msgs, a.max_tokens, a.timeout, system=False)
    res.add(
        "long",
        str(code) in text,
        f"{tm.get('prompt_tokens')} tokens, needle at {at}/{n_fill}: {text!r} ({dt:.0f} s, {tm.get('steps')} steps)",
    )


# ------------------------------------------------------------------ creative suite
NAMES = [
    "Alder",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fern",
    "Gorse",
    "Hazel",
    "Iris",
    "Juniper",
    "Kale",
    "Larch",
    "Maple",
    "Nettle",
    "Oak",
    "Poppy",
    "Quince",
    "Rowan",
    "Sage",
    "Thyme",
    "Umber",
    "Violet",
    "Willow",
    "Yarrow",
]
WORDS = [
    "harbor",
    "quartz",
    "lantern",
    "meadow",
    "violin",
    "glacier",
    "pepper",
    "orbit",
    "saddle",
    "tundra",
    "falcon",
    "marble",
    "ember",
    "canyon",
    "biscuit",
    "nectar",
    "pylon",
    "walnut",
    "zephyr",
    "kettle",
    "basalt",
    "juniper",
]
UNICODE = ["日本語のテキスト", "Ωμέγα και άλφα", "ñandú y pingüino", "🚀🌍✨", "Ελλάδα 2026", "東京タワー", "naïve café"]


def nonce(rng):
    """A random tag at the start of a prompt: a fresh prefix, so the case is not served from another case's cache."""
    return f"[session {rng.randint(10**7, 10**8 - 1)}] "


def case_chain(a, rng, res):
    facts = {
        "name": rng.choice(["Mira", "Tomas", "Ilse", "Ravi", "Noor", "Kenji", "Lucia", "Oskar"]),
        "city": rng.choice(["Lisbon", "Osaka", "Tallinn", "Cusco", "Hobart", "Ghent", "Bergen", "Fez"]),
        "pet": rng.choice(["parrot", "tortoise", "ferret", "goldfish", "beagle", "iguana"]),
        "number": str(rng.randint(10, 99)),
    }
    turns = [
        f"{nonce(rng)}Hi! My name is {facts['name']}. Just say hello back.",
        f"I live in {facts['city']}. Acknowledge briefly.",
        f"I have a {facts['pet']} at home. Acknowledge briefly.",
        f"My lucky number is {facts['number']}. Acknowledge briefly.",
        "What is 2 plus 2? Answer with the number only.",
        "Now tell me my name, my city, my pet and my lucky number, separated by commas.",
    ]
    msgs, out, res0 = [], "", []
    for t in turns:
        msgs.append({"role": "user", "content": t})
        out, _, tm, _ = chat(a.url, msgs, 32, a.timeout)
        res0.append((tm.get("first_step") or {}).get("resident"))
        msgs.append({"role": "assistant", "content": out})
    miss = [k for k, v in facts.items() if v.lower() not in out.lower()]
    res.add("chain", not miss, f"{facts} -> {out!r} (missing {miss}; residents per turn {res0})", "model")


def case_retrieve(a, rng, res):
    n_rec = rng.randint(100, 1500)
    keys = rng.sample(range(10000, 99999), n_rec)
    recs = {f"{rng.choice(NAMES)}-{k}": rng.randint(100000, 999999) for k in keys}
    items = list(recs.items())
    lines = []
    for i, (k, v) in enumerate(items):
        lines.append(f"Record {k}: {v}.")
        if rng.random() < 0.3:
            lines.append(FILLER.strip())
    ask = rng.sample(items, 3)
    q = (
        f"{nonce(rng)}Below is a register of records.\n\n"
        + "\n".join(lines)
        + "\n\nGive the numbers of records "
        + ", ".join(k for k, _ in ask)
        + ", in that order, separated by spaces. Numbers only."
    )
    text, _, tm, dt = chat(a.url, [{"role": "user", "content": q}], 32, a.timeout, system=False)
    got = [str(v) in text for _, v in ask]
    res.add(
        "retrieve",
        all(got),
        f"{tm.get('prompt_tokens')} tokens, {n_rec} records, want " f"{[v for _, v in ask]} -> {text!r} ({dt:.0f} s)",
        "model",
    )


def case_copy(a, rng, res):
    nums = [str(rng.randint(100, 999)) for _ in range(40)]
    q = f"{nonce(rng)}Repeat this list exactly, separated by single spaces, and nothing else:\n" + " ".join(nums)
    text, ids, tm, dt = chat(a.url, [{"role": "user", "content": q}], 220, a.timeout)
    got = text.replace(",", " ").split()
    ok = got[: len(nums)] == nums
    k = next((i for i, (x, y) in enumerate(zip(got, nums)) if x != y), min(len(got), len(nums)))
    res.add(
        "copy",
        ok,
        f"{len(ids)} tokens, {'exact' if ok else f'first difference at item {k}'} ({dt:.0f} s, "
        f"{tm.get('steps')} steps)",
        "model",
    )


def sized_prompt(a, rng, target, tail):
    """nonce + FILLER x n + " ok" x m + tail, measured through /v1/completions (usage.prompt_tokens, max_tokens 1)
    until it is exactly ``target`` tokens (or as close as the tokenizer allows). The probes share the final prompt's
    prefix, so the engine serves most of them from its cache."""
    head = nonce(rng)

    def build(n, m):
        return head + FILLER * n + " ok" * m + tail

    def measure(p):
        return post(f"{a.url}/v1/completions", {"prompt": p, "max_tokens": 1}, a.timeout)["usage"]["prompt_tokens"]

    t20, t40 = measure(build(20, 0)), measure(build(40, 0))
    per = (t40 - t20) / 20
    n = max(1, int((target - (t20 - 20 * per)) / per) - 1)
    m = 0
    for _ in range(3):
        t = measure(build(n, m))
        if t == target:
            break
        m = max(0, m + target - t)
    return build(n, m), t


def case_boundary(a, rng, res):
    """A prompt ending 1-48 tokens before a chunk edge (5120 * k); 64 new tokens cross it."""
    edge = 5120 * rng.randint(1, 3)
    code = rng.randint(100000, 999999)
    tail = f"\n\nThe launch code is {code}. Repeat the launch code five times, separated by spaces, then write DONE."
    prompt, _ = sized_prompt(a, rng, edge - rng.randint(1, 48), tail)
    r = post(f"{a.url}/v1/completions", {"prompt": prompt, "max_tokens": 64, "temperature": 0}, a.timeout)
    text, u = r["choices"][0]["text"], r["usage"]
    crossed = u["prompt_tokens"] < edge <= u["prompt_tokens"] + u["completion_tokens"]
    res.add(
        "boundary",
        str(code) in text,
        f"prompt {u['prompt_tokens']} (edge {edge}, crossed {crossed}, " f"+{u['completion_tokens']}): {text[:70]!r}",
        "model",
    )


def case_math(a, rng, res):
    x, y, z, w = (rng.randint(2, 30) for _ in range(4))
    want = (x + y) * z - w
    text, _, _, _ = chat(
        a.url,
        [{"role": "user", "content": f"{nonce(rng)}Compute ({x} + {y}) * {z} - {w}. " "Answer with the number only."}],
        16,
        a.timeout,
    )
    res.add("math", str(want) in text.replace(",", ""), f"({x}+{y})*{z}-{w} = {want} -> {text!r}", "model")


def case_json(a, rng, res):
    want = {"name": rng.choice(NAMES), "age": rng.randint(18, 90), "city": rng.choice(["Oslo", "Lima", "Pune"])}
    q = (
        f"{nonce(rng)}Output only a JSON object with the keys name, age and city for this person: "
        f"{want['name']}, {want['age']} years old, lives in {want['city']}. No code fences, no other text."
    )
    text, _, _, _ = chat(a.url, [{"role": "user", "content": q}], 48, a.timeout)
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        got = json.loads(t)
        ok = {k: got.get(k) for k in want} == want
    except json.JSONDecodeError:
        ok = False
    res.add("json", ok, f"{want} -> {text!r}", "model")


def case_sort(a, rng, res):
    ws = rng.sample(WORDS, 8)
    text, _, _, _ = chat(
        a.url,
        [
            {
                "role": "user",
                "content": f"{nonce(rng)}Sort these words alphabetically and output "
                f"them separated by single spaces, nothing else: {' '.join(ws)}",
            }
        ],
        40,
        a.timeout,
    )
    got = [w.strip(".,").lower() for w in text.split()]
    res.add("sort", got == sorted(ws), f"{ws} -> {text!r}", "model")


def case_unicode(a, rng, res):
    s = " ".join(rng.sample(UNICODE, 3))
    text, _, _, _ = chat(
        a.url,
        [
            {
                "role": "user",
                "content": f"{nonce(rng)}Repeat exactly the text between the " f"brackets, without the brackets: [{s}]",
            }
        ],
        48,
        a.timeout,
    )
    res.add("unicode", s in text, f"{s!r} -> {text!r}", "model")


def case_thinking(a, rng, res):
    x, y = rng.randint(11, 49), rng.randint(11, 49)
    text, _, tm, dt = chat(
        a.url,
        [{"role": "user", "content": f"{nonce(rng)}What is {x} * {y}? Give the final " "number."}],
        512,
        a.timeout,
        system=False,  # SYSTEM says: no reasoning
        chat_template_kwargs={"enable_thinking": True},
    )
    ok = bool(tm["reasoning"]) and str(x * y) in text.replace(",", "")
    res.add(
        "thinking",
        ok,
        f"{x}*{y}={x * y}: reasoning {len(tm['reasoning'])} chars, finish {tm['finish']}, "
        f"answer {text[-80:]!r} ({dt:.0f} s)",
        "model",
    )


def case_raw(a, rng, res):
    p, want = rng.choice(
        [
            ("The capital of Italy is", "Rome"),
            ("Water freezes at a temperature of 0 degrees", "Celsius"),
            ("The first president of the United States was George", "Washington"),
        ]
    )
    r = post(f"{a.url}/v1/completions", {"prompt": p, "max_tokens": 8, "temperature": 0}, a.timeout)
    text = r["choices"][0]["text"]
    res.add("raw", want.lower() in text.lower(), f"{p!r} -> {text!r}", "model")


def case_seeded(a, rng, res):
    q = f"{nonce(rng)}Write one sentence about {rng.choice(WORDS)}s."
    seed = rng.randint(0, 2**31 - 1)
    kw = {"temperature": 0.9, "top_p": 0.95, "seed": seed}
    t1, i1, _, _ = chat(a.url, [{"role": "user", "content": q}], 24, a.timeout, **kw)
    t2, i2, _, _ = chat(a.url, [{"role": "user", "content": q}], 24, a.timeout, **kw)
    k = next((i for i, (x, y) in enumerate(zip(i1, i2)) if x != y), None)
    res.add("seeded", i1 == i2, f"seed {seed}: {'identical' if i1 == i2 else f'diverged at {k}'} {t1!r} / {t2!r}")


def case_twins(a, rng, res):
    q = (
        f"{nonce(rng)}Name two {rng.choice(['rivers', 'mountains', 'deserts', 'islands'])} in "
        f"{rng.choice(['Asia', 'Africa', 'Europe', 'South America'])} and one fact about each."
    )
    out = [None, None]

    def go(i):
        out[i] = chat(a.url, [{"role": "user", "content": q}], 24, a.timeout)

    th = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if None in out:
        res.add("twins", False, "a twin raised (see the HTTP error above)")
        return
    (t1, i1, m1, _), (t2, i2, m2, _) = out
    r = [(m.get("first_step") or {}).get(k) for m in (m1, m2) for k in ("slot", "resident")]
    k = next((i for i, (x, y) in enumerate(zip(i1, i2)) if x != y), None)
    res.add("twins", i1 == i2, f"{'identical' if i1 == i2 else f'diverged at {k}'} (slot/resident {r}): {t1[:60]!r}")


def case_stop(a, rng, res):
    n = rng.randint(4, 9)
    text, _, tm, _ = chat(
        a.url,
        [{"role": "user", "content": f"{nonce(rng)}Count from 1 to 12, separated by spaces. " "Numbers only."}],
        40,
        a.timeout,
        stop=[f" {n} "],
    )
    nums = text.split()
    ok = tm["finish"] == "stop" and str(n) not in nums and nums[-1:] == [str(n - 1)]
    res.add("stop", ok, f"stop ' {n} ' -> {text!r}, finish {tm['finish']}")


def case_cancel(a, rng, res):
    host, port = a.url.split("//")[1].split(":") if ":" in a.url.split("//")[1] else (a.url.split("//")[1], "80")
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": f"{nonce(rng)}Write a long story about a lighthouse."}],
            "max_tokens": 400,
            "stream": True,
        }
    ).encode()
    c = http.client.HTTPConnection(host, int(port), timeout=a.timeout)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json", "X-Xing-Pool": "hammer"})
    r = c.getresponse()
    seen = 0
    while seen < 3:
        line = r.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            seen += 1
    c.close()  # the server notices on its next write and cancels the generation
    text, _, tm, _ = chat(a.url, [{"role": "user", "content": f"{nonce(rng)}Say OK."}], 8, a.timeout)
    res.add("cancel", seen == 3 and bool(text), f"closed after {seen} chunks; next request -> {text!r}")


def case_oversize(a, rng, res):
    prompt = nonce(rng) + FILLER * 1100 + "\n\nSummarize."  # ~67K tokens
    try:
        post(f"{a.url}/v1/completions", {"prompt": prompt, "max_tokens": 8}, a.timeout)
        res.add("oversize", False, "a ~60K-token prompt was accepted")
    except urllib.error.HTTPError as e:
        res.add("oversize", e.code == 400, f"HTTP {e.code}: {e.read()[:120]!r}")


def case_maxctx(a, rng, res):
    prompt, n = sized_prompt(a, rng, 56320 - rng.randint(8, 40), "\n\nNow describe the committee meeting in detail.")
    r = post(f"{a.url}/v1/completions", {"prompt": prompt, "max_tokens": 400, "temperature": 0}, a.timeout)
    u, fin = r["usage"], r["choices"][0]["finish_reason"]
    ok = fin == "length" and u["prompt_tokens"] + u["completion_tokens"] == 56320
    res.add(
        "maxctx",
        ok,
        f"prompt {u['prompt_tokens']} + {u['completion_tokens']} = "
        f"{u['prompt_tokens'] + u['completion_tokens']}, finish {fin}",
    )


CREATIVE = {  # case: weight (how often a worker picks it)
    case_fact: 2,
    case_memory: 2,
    case_repeat: 2,
    case_chain: 2,
    case_retrieve: 3,
    case_copy: 2,
    case_boundary: 2,
    case_math: 2,
    case_json: 2,
    case_sort: 2,
    case_unicode: 2,
    case_thinking: 1,
    case_raw: 2,
    case_seeded: 2,
    case_twins: 2,
    case_stop: 2,
    case_cancel: 1,
    case_oversize: 1,
    case_maxctx: 1,
    case_long: 1,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--rounds", type=int, default=1, help="rounds of every case per worker (ignored with --duration-min)"
    )
    ap.add_argument("--duration-min", type=float, default=0, help="run random cases until this many minutes pass")
    ap.add_argument("--long", type=int, default=0, help="add the long case with a ~N-token prompt (0: off)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-system", action="store_true", help="no system turn: short prompts go cold every step")
    ap.add_argument("--suite", choices=("default", "creative"), default="default")
    a = ap.parse_args()
    global SYSTEM_ON
    SYSTEM_ON = not a.no_system
    a.url = a.url.rstrip("/")
    if a.suite == "creative":
        a.long = a.long or 30000
        cases, weights = list(CREATIVE), list(CREATIVE.values())
    else:
        cases = [case_fact, case_memory, case_repeat] + ([case_long] if a.long else [])
        weights = [1] * len(cases)
    res = Results()
    t_end = time.time() + a.duration_min * 60

    def worker(w):
        rng = random.Random(a.seed * 1000 + w)
        i = 0
        while True:
            if a.duration_min:
                if time.time() > t_end:
                    return
                fn = rng.choices(cases, weights)[0]
            else:
                if i >= a.rounds * len(cases):
                    return
                fn = cases[(i + w) % len(cases)]
            i += 1
            try:
                fn(a, rng, res)
            except urllib.error.HTTPError as e:
                res.add(fn.__name__[5:], False, f"HTTP {e.code}: {e.read()[:300]!r}")
            except Exception as e:
                res.add(fn.__name__[5:], False, f"{type(e).__name__}: {e}")

    t0 = time.time()
    th = [threading.Thread(target=worker, args=(w,)) for w in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    try:
        with urllib.request.urlopen(f"{a.url}/stats", timeout=60) as r:
            stats = json.loads(r.read())
    except Exception as e:
        stats = {"error": str(e)}
    srv = stats.get("server", {})
    print(f"\n[hammer] {time.time() - t0:.0f} s, concurrency {a.concurrency}, suite {a.suite}")
    for c, (p, f) in sorted(res.by_case.items()):
        print(f"  {c:9s} pass {p}  fail {f}")
    print(
        f"  server: steps {srv.get('steps')}, tokens {srv.get('tokens')}, remounts {srv.get('remounts')}, cold "
        f"{srv.get('cold')}, step p50 {srv.get('step_s_p50')} s p99 {srv.get('step_s_p99')} s, check_fail "
        f"{srv.get('check_fail')}, engine_errors {srv.get('engine_errors')}, heartbeat_mismatch "
        f"{srv.get('heartbeat_mismatch')}"
    )
    print(f"  engine: {json.dumps((stats.get('engine') or {}).get('telemetry'))}")
    for f in res.fails + srv.get("recent_failures", []):
        print(f"  FAIL {f}")
    for f in res.model_fails:
        print(f"  MISS (model) {f}")
    bad = res.fails or srv.get("check_fail") or srv.get("engine_errors") or "error" in stats
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
