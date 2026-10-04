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
Exit 1 if any case failed or the server counted a check failure.
"""

from __future__ import annotations

import argparse
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


def chat(base, messages, max_tokens, timeout, system=True):
    t0 = time.time()
    if system and SYSTEM_ON:
        messages = [{"role": "system", "content": SYSTEM}] + messages
    r = post(f"{base}/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, "temperature": 0}, timeout)
    ch = r["choices"][0]
    return ch["message"].get("content") or "", ch.get("token_ids", []), r.get("timings", {}), time.time() - t0


class Results:
    def __init__(self):
        self.lock = threading.Lock()
        self.by_case = {}
        self.fails = []

    def add(self, case, ok, detail):
        with self.lock:
            p, f = self.by_case.get(case, (0, 0))
            self.by_case[case] = (p + ok, f + (not ok))
            if not ok:
                self.fails.append(f"{case}: {detail}")
        print(f"[hammer] {time.strftime('%H:%M:%S')} {'ok  ' if ok else 'FAIL'} {case}: {detail}", flush=True)


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
    a = ap.parse_args()
    global SYSTEM_ON
    SYSTEM_ON = not a.no_system
    a.url = a.url.rstrip("/")
    cases = [case_fact, case_memory, case_repeat] + ([case_long] if a.long else [])
    res = Results()
    t_end = time.time() + a.duration_min * 60

    def worker(w):
        rng = random.Random(a.seed * 1000 + w)
        i = 0
        while True:
            if a.duration_min:
                if time.time() > t_end:
                    return
                fn = rng.choice(cases)
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
    print(f"\n[hammer] {time.time() - t0:.0f} s, concurrency {a.concurrency}")
    for c, (p, f) in sorted(res.by_case.items()):
        print(f"  {c:8s} pass {p}  fail {f}")
    print(
        f"  server: steps {srv.get('steps')}, tokens {srv.get('tokens')}, remounts {srv.get('remounts')}, cold "
        f"{srv.get('cold')}, step p50 {srv.get('step_s_p50')} s p99 {srv.get('step_s_p99')} s, check_fail "
        f"{srv.get('check_fail')}, engine_errors {srv.get('engine_errors')}, heartbeat_mismatch "
        f"{srv.get('heartbeat_mismatch')}"
    )
    print(f"  engine: {json.dumps((stats.get('engine') or {}).get('telemetry'))}")
    for f in res.fails + srv.get("recent_failures", []):
        print(f"  FAIL {f}")
    bad = res.fails or srv.get("check_fail") or srv.get("engine_errors") or "error" in stats
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
