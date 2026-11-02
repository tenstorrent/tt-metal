# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Client and load generator of the System One server.

python -m models.demos.strands_agent.serve.client --wait-ready 600 --save out.json
python -m models.demos.strands_agent.serve.client --load 200 --concurrency 16 --route --tokens 512
python -m models.demos.strands_agent.serve.client --load 2000 --rate 100 --route --tokens 512
python -m models.demos.strands_agent.serve.client --load 3000 --concurrency 128 --route --tokens 128 --procs 4"""

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

EXAMPLES_PATH = Path(__file__).parent.parent / "tests" / "examples.json"
ROUTE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "payments, invoices, refunds and charges",
            "sales": "new purchases, quotes and pricing",
            "retail": "in-store orders, shipping and returns",
            "support": "product help and troubleshooting",
        },
    }
}
STATE_TEXT = (
    "I was charged twice for my order #48211 this month and nobody has answered my emails for three days. "
    "Please fix this or I will dispute both charges with my bank. "
)


def load_examples(only=None):
    from models.demos.strands_agent.tests.common import load_example_state
    from models.demos.strands_agent.tt.weight_mapping import resolve_checkpoint

    examples = json.loads(EXAMPLES_PATH.read_text())
    ckpt_dir = resolve_checkpoint() if any("state_file" in e for e in examples) else None
    out = {}
    for ex in examples:
        if only is None or ex["name"] == only:
            out[ex["name"]] = {"state": load_example_state(ex, ckpt_dir), "questions": ex["questions"]}
    if not out:
        raise SystemExit(f"no example named {only!r}")
    return out


def route_request(target_tokens):
    """Route question over a state of about target_tokens tokens (word count bisected like tests/test_perf.py)."""
    from transformers import AutoTokenizer

    from models.demos.strands_agent.tt.head import load_config
    from models.demos.strands_agent.tt.prompting import encode_request
    from models.demos.strands_agent.tt.weight_mapping import resolve_checkpoint

    ckpt_dir = resolve_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    max_length = load_config(ckpt_dir).max_length
    words = STATE_TEXT.split()

    def state_for(count):
        return " ".join((words * (count // len(words) + 1))[:count])

    def tokens(count):
        return len(encode_request(tokenizer, state_for(count), ROUTE_QUESTIONS, max_length)[0].input_ids)

    lo, hi = 1, 4 * target_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if tokens(mid) >= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    return {"state": state_for(lo), "questions": ROUTE_QUESTIONS}


def compact(answer):
    if answer["type"] == "noul":
        return f"noul {answer['noul']:.3f}"
    if answer["type"] == "choice":
        return f"choice {answer['choice']} (conf {answer['confidence']:.2f})"
    return f"score {answer['score']:.2f}"


def pct(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


async def wait_ready(client, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            r = await client.get("/health")
            body = r.json()
            if body.get("status") == "ok":
                print("server ready")
                return
            if body.get("status") == "failed":
                print(f"server failed: {body.get('error')}")
                sys.exit(1)
            print(f"waiting ({int(deadline - time.monotonic())} s left): {body.get('status')}", flush=True)
        except httpx.HTTPError as e:
            print(f"waiting ({int(deadline - time.monotonic())} s left): {type(e).__name__}", flush=True)
        if time.monotonic() >= deadline:
            print("timeout waiting for the server")
            sys.exit(1)
        await asyncio.sleep(2)


async def run_examples(client, args):
    responses = {}
    for name, req in load_examples(args.example).items():
        r = await client.post("/v1/systemone", json=req)
        r.raise_for_status()
        body = r.json()
        responses[name] = body
        print(f"{name}: {body['latency_ms']} ms, usage {body['usage']}")
        for q, a in body["answers"].items():
            print(f"  {q}: {compact(a)}")
    if args.save:
        Path(args.save).write_text(json.dumps(responses, indent=2))
        print(f"saved {args.save}")


async def load_worker(req, load, concurrency, rate, seed, url, timeout):
    """One load generator: send `load` requests; returns raw latencies (ms), error counts and wall start/end."""
    latencies, errors = [], {}
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4, keepalive_expiry=5.0
    )

    def fail(key):
        errors[key] = errors.get(key, 0) + 1

    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) as client:

        async def one():
            t = time.perf_counter()
            try:
                r = await client.post("/v1/systemone", json=req)
            except Exception as e:
                fail(type(e).__name__)
                return
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t) * 1000)
            else:
                fail(f"http {r.status_code}")

        async def guarded():
            async with sem:
                await one()

        t_start = time.time()
        t0 = time.perf_counter()
        if rate > 0:
            rng = random.Random(seed)
            tasks, due = [], 0.0
            for _ in range(load):
                due += rng.expovariate(rate)
                delay = t0 + due - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(guarded()))
            await asyncio.gather(*tasks)
        else:
            await asyncio.gather(*(guarded() for _ in range(load)))
        t_end = time.time()
    return {"latencies": latencies, "errors": errors, "t_start": t_start, "t_end": t_end}


def load_child(k, queue, *worker_args):
    queue.put((k, asyncio.run(load_worker(*worker_args))))


def split_load(args):
    """Per-process (load, concurrency, rate, seed) for --procs P: load N/P (remainder to the first), concurrency C/P
    (min 1), rate R/P, seed + k."""
    p = args.procs
    return [
        (
            args.load // p + (1 if k < args.load % p else 0),
            max(1, args.concurrency // p),
            args.rate / p,
            args.seed + k,
        )
        for k in range(p)
    ]


def run_children(req, args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=load_child, args=(k, queue, req, load, conc, rate, seed, args.url, args.timeout))
        for k, (load, conc, rate, seed) in enumerate(split_load(args))
    ]
    for pr in procs:
        pr.start()
    results = dict(queue.get() for _ in procs)
    for pr in procs:
        pr.join()
    return [results[k] for k in range(len(procs))]


async def run_load(client, args):
    if args.route:
        req = route_request(args.tokens)
    else:
        examples = load_examples()
        req = examples[args.example] if args.example else next(iter(examples.values()))
    if args.questions > 1:
        name, spec = next(iter(req["questions"].items()))
        req = {**req, "questions": {f"{name}_{i}": spec for i in range(args.questions)}}
    n_questions = len(req["questions"])
    (await client.post("/v1/systemone", json=req)).raise_for_status()
    await client.get("/metrics", params={"reset": "true"})
    open_loop = args.rate > 0
    if args.procs == 1:
        load, conc, rate, seed = split_load(args)[0]
        results = [await load_worker(req, load, conc, rate, seed, args.url, args.timeout)]
    else:
        results = await asyncio.get_running_loop().run_in_executor(None, run_children, req, args)
    latencies = [x for r in results for x in r["latencies"]]
    errors = {}
    for r in results:
        for key, n in r["errors"].items():
            errors[key] = errors.get(key, 0) + n
    wall = max(r["t_end"] for r in results) - min(r["t_start"] for r in results)
    done, n_err = len(latencies), sum(errors.values())
    mode = f"open loop at {args.rate:g} req/s offered" if open_loop else f"concurrency {args.concurrency}"
    if args.procs > 1:
        mode += f", {args.procs} client processes"
    print(f"{done}/{args.load} ok ({n_err} errors) in {wall:.2f} s, {mode}")
    if errors:
        print("errors:", json.dumps(errors))
    print(f"{done / wall:.1f} req/s, {done * n_questions / wall:.1f} questions/s ({n_questions} per request)")
    if latencies:
        print(
            f"client latency ms: p50 {pct(latencies, 50):.1f} p90 {pct(latencies, 90):.1f} "
            f"p99 {pct(latencies, 99):.1f} max {max(latencies):.1f} mean {statistics.mean(latencies):.1f}"
        )
    print("server metrics:", json.dumps((await client.get("/metrics")).json(), indent=2))
    if args.load and not done:
        sys.exit(1)


async def main_async(args):
    args.concurrency = args.concurrency or (4096 if args.rate > 0 else 8)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=5.0)
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        if args.wait_ready:
            await wait_ready(client, args.wait_ready)
        if args.load:
            await run_load(client, args)
        elif not args.wait_ready or args.save or args.example:
            await run_examples(client, args)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--save", help="write {example: response} JSON here")
    p.add_argument("--example", help="example name (default: all in examples mode, the first in load mode)")
    p.add_argument("--load", type=int, default=0, help="send N requests")
    p.add_argument("--concurrency", type=int, default=None, help="in-flight cap (default 8; 4096 with --rate)")
    p.add_argument("--rate", type=float, default=0, help="open loop with --load: Poisson arrivals at R requests/s")
    p.add_argument("--seed", type=int, default=0, help="open loop arrival seed")
    p.add_argument("--procs", type=int, default=1, help="load mode: split the load over P client processes")
    p.add_argument("--route", action="store_true", help="load mode: route question over a state of --tokens tokens")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--questions", type=int, default=1, help="load mode: K copies of the first question per request")
    p.add_argument("--wait-ready", type=float, default=0, metavar="SECONDS")
    p.add_argument("--timeout", type=float, default=120)
    args = p.parse_args(argv)
    if args.rate and not args.load:
        p.error("--rate needs --load")
    if args.procs < 1:
        p.error("--procs must be >= 1")
    if args.questions < 1:
        p.error("--questions must be >= 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
