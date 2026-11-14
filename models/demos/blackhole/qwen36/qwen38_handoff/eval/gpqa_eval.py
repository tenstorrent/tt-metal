#!/usr/bin/env python3
"""GPQA-Diamond zero-shot eval client for an OpenAI-compatible server (vLLM)."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time

TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
ANS_RE = re.compile(r"(?i)Answer\s*:\s*\**\s*\$?\(?([A-D])\)?")
DEFAULT_CSV = "/home/ttuser/atupe/qwen38_work/tier2/gpqa/gpqa_diamond.csv"


def extract_answer(text):
    if not text:
        return None
    m = ANS_RE.findall(text)
    return m[-1].upper() if m else None


def sampling_config(args):
    """Sampling/extraction config as recorded in the summary (None = not sent)."""
    return {
        "temperature": args.temperature,
        "top_p": args.top_p if args.top_p >= 0 else None,
        "top_k": args.top_k if args.top_k > 0 else None,
        "max_tokens": args.max_tokens if args.max_tokens > 0 else None,
        "presence_penalty": getattr(args, "presence_penalty", None),
        "no_think": args.no_think,
        "final_only": args.final_only,
        "extra_body": json.loads(args.extra_body) if args.extra_body else None,
    }


def build_request(args):
    """Return (top-level kwargs, extra_body) for chat.completions.create; omits disabled params."""
    kw = {"temperature": args.temperature}
    if args.top_p >= 0:
        kw["top_p"] = args.top_p
    if args.max_tokens > 0:
        kw["max_tokens"] = args.max_tokens
    extra = {}
    if args.top_k > 0:
        extra["top_k"] = args.top_k
    if args.no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if args.extra_body:
        extra.update(json.loads(args.extra_body))  # after built-ins so it can override
    return kw, extra


def extract_with_source(extract, content, reasoning, final_only=False, allow_fallback=True):
    """Return (answer, source) with source in "content" | "reasoning" | None."""
    ext = extract(content)
    if ext is not None:
        return ext, "content"
    if final_only or not allow_fallback or not reasoning:
        return None, None
    ext = extract(reasoning + "\n" + content)
    return ext, ("reasoning" if ext is not None else None)


def add_sampling_args(p):
    p.add_argument(
        "--final-only", action="store_true", help="extract answer only from final content (no reasoning fallback)"
    )
    p.add_argument(
        "--extra-body", default=None, help="JSON object merged into request extra_body (overrides built-ins)"
    )


def load_questions(path, shuffle_seed=0):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 198, f"expected 198 rows, got {len(rows)}"
    qs = []
    for r in rows:
        rid = r["Record ID"]
        choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"{rid}-{shuffle_seed}").shuffle(order)
        shuffled = [choices[i] for i in order]
        correct = "ABCD"[order.index(0)]
        prompt = TEMPLATE.format(question=r["Question"], A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3])
        qs.append({"record_id": rid, "prompt": prompt, "correct_letter": correct})
    return qs


def stable_seed(rid, offset=0):
    return (int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) + offset) % (2**31 - 1)


def trunc(t, n=2000):
    if t is None:
        return ""
    return t if len(t) <= 2 * n else t[:n] + "\n...[TRUNCATED]...\n" + t[-n:]


async def run_one(client, model, q, args):
    req, extra = build_request(args)
    last_err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q["prompt"]}],
                **req,
                seed=stable_seed(q["record_id"], args.seed_offset),
                extra_body=extra,
            )
            ch = resp.choices[0]
            msg = ch.message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            ext, src = extract_with_source(extract_answer, content, reasoning, args.final_only)
            u = resp.usage
            rt = None
            det = getattr(u, "completion_tokens_details", None) if u else None
            if det is not None:
                rt = getattr(det, "reasoning_tokens", None)
            return {
                "record_id": q["record_id"],
                "correct_letter": q["correct_letter"],
                "extracted": ext,
                "extracted_from": src,
                "correct": ext == q["correct_letter"],
                "finish_reason": ch.finish_reason,
                "completion_tokens": u.completion_tokens if u else None,
                "reasoning_tokens": rt,
                "latency_s": round(time.time() - t0, 3),
                "response_text": trunc((reasoning + "\n</think>\n" + content) if reasoning else content),
            }
        except Exception as e:  # connection, 5xx, etc.
            last_err = f"{type(e).__name__}: {e}"
            if attempt < 3:
                await asyncio.sleep(2**attempt * args.backoff)
    return {
        "record_id": q["record_id"],
        "correct_letter": q["correct_letter"],
        "extracted": None,
        "extracted_from": None,
        "correct": False,
        "error": last_err,
        "finish_reason": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "latency_s": None,
        "response_text": "",
    }


def summarize(path, wall, config=None):
    recs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["record_id"]] = r
    ok = [r for r in recs.values() if not r.get("error")]
    errs = [r for r in recs.values() if r.get("error")]
    toks = sorted(r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None)

    def pct(p):
        return toks[min(len(toks) - 1, int(round(p * (len(toks) - 1))))] if toks else None

    s = {
        "n": len(ok),
        "n_errors": len(errs),
        "accuracy": (sum(r["correct"] for r in ok) / len(ok)) if ok else None,
        "n_correct": sum(r["correct"] for r in ok),
        "n_no_answer": sum(r["extracted"] is None for r in ok),
        "n_from_reasoning": sum(r.get("extracted_from") == "reasoning" for r in ok),
        "n_finish_length": sum(r.get("finish_reason") == "length" for r in ok),
        "mean_completion_tokens": statistics.mean(toks) if toks else None,
        "p50_completion_tokens": pct(0.5),
        "p95_completion_tokens": pct(0.95),
        "wall_time_s_this_run": round(wall, 1),
    }
    if config is not None:
        s["sampling_config"] = config
    with open(path + ".summary.json", "w") as f:
        json.dump(s, f, indent=2)
    return s


async def amain(args):
    from openai import AsyncOpenAI

    qs = load_questions(args.csv, args.shuffle_seed)
    if args.ids:
        want = {l.strip() for l in open(args.ids) if l.strip()}
        qs = [q for q in qs if q["record_id"] in want]
    if args.limit:
        qs = qs[: args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["record_id"])
                    except Exception:
                        pass
    todo = [q for q in qs if q["record_id"] not in done]
    print(
        f"{len(qs)} selected, {len(done & {q['record_id'] for q in qs})} already done, {len(todo)} to run", flush=True
    )
    t0 = time.time()
    if todo:
        client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout, max_retries=0)
        model = args.model or (await client.models.list()).data[0].id
        print(f"model: {model}", flush=True)
        sem = asyncio.Semaphore(args.concurrency)
        n = nc = 0
        total = len(todo)
        lock = asyncio.Lock()
        fout = open(args.out, "a")

        async def worker(q):
            nonlocal n, nc
            async with sem:
                r = await run_one(client, model, q, args)
            async with lock:
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                n += 1
                nc += bool(r["correct"])
                if n % 10 == 0 or n == total:
                    print(f"[{n}/{total}] running_acc={nc / n:.3f} elapsed={time.time() - t0:.0f}s", flush=True)

        await asyncio.gather(*(worker(q) for q in todo))
        fout.close()
        await client.close()
    s = summarize(args.out, time.time() - t0, sampling_config(args)) if os.path.exists(args.out) else {}
    print(json.dumps(s, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="default: first from GET /v1/models")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None, help="file of record ids, one per line")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95, help="<0: do not send top_p")
    p.add_argument("--top-k", type=int, default=20, help="<=0: do not send top_k")
    p.add_argument("--max-tokens", type=int, default=32768, help="0: do not send max_tokens")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--backoff", type=float, default=2.0)
    add_sampling_args(p)
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
