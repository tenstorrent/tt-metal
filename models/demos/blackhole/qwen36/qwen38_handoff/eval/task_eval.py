#!/usr/bin/env python3
"""General task-level eval client (gpqa | mmlupro | aime) for an OpenAI-compatible server (vLLM).

Same design as gpqa_eval.py (async concurrency, resume, JSONL, summary, seeds, retries, reasoning/content handling).
Defaults follow the Qwen3.8 card: temp 1.0, top_p 0.95, top_k 20, presence_penalty 0, thinking on,
and NO reasoning_effort is sent (the chat template default, xhigh, applies).
--repeats k writes k files <out stem>.r0.jsonl ... .r{k-1}.jsonl with seed offsets seed_offset+0..k-1.
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpqa_eval as g  # GPQA prompt/extraction/shuffle reused verbatim

DATA = "/home/ttuser/atupe/qwen38_work/tier2/data"
MMLU_JSONL = f"{DATA}/mmlu_pro_test.jsonl"
MMLU_IDS = f"{DATA}/mmlu_pro_500_ids.txt"
AIME_JSONL = f"{DATA}/aime_2026.jsonl"

MMLU_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. "
    "Think step by step before answering.\n\n{question}\n\n{options}"
)
MMLU_RE = re.compile(r"(?i)Answer\s*:\s*\**\s*\$?\(?([A-J])\)?")
AIME_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_letter_mmlu(text):
    if not text:
        return None
    m = MMLU_RE.findall(text)
    return m[-1].upper() if m else None


def last_boxed(text):
    """Content of the last \\boxed{...} (or \\fbox{...}) with balanced braces; None if absent/unbalanced."""
    if not text:
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def normalize_int(s):
    """Normalize a boxed string to an integer 0-999, else None."""
    if s is None:
        return None
    t = s.strip()
    t = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace("\\;", "").replace("\\%", "")
    t = t.replace("$", "").replace("{,}", "").replace(",", "").replace(" ", "").replace("\u00a0", "")
    t = t.rstrip(".")
    m = re.fullmatch(r"\+?(\d+)(?:\.0+)?", t)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 999 else None


def extract_aime(text):
    return normalize_int(last_boxed(text))


# ---------------- task definitions: each returns questions [{id, prompt, gold}] ----------------
def load_gpqa(args):
    qs = g.load_questions(args.csv or g.DEFAULT_CSV, args.shuffle_seed)
    return [{"id": q["record_id"], "prompt": q["prompt"], "gold": q["correct_letter"]} for q in qs]


def load_mmlupro(args):
    rows = [json.loads(l) for l in open(args.data or MMLU_JSONL) if l.strip()]
    ids_path = args.ids if args.ids is not None else MMLU_IDS
    if ids_path != "all":
        want = [int(l) for l in open(ids_path) if l.strip()]
        ws = set(want)
        rows = [r for r in rows if r["question_id"] in ws]
    qs = []
    for r in rows:
        n = len(r["options"])
        letters = "ABCDEFGHIJ"[:n]
        opts = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(r["options"]))
        qs.append(
            {
                "id": str(r["question_id"]),
                "gold": r["answer"],
                "category": r["category"],
                "prompt": MMLU_TEMPLATE.format(letters="ABCDEFGHIJ", question=r["question"], options=opts),
            }
        )
    random.Random(0).shuffle(qs)  # deterministic mixed order so --limit gives a category mix
    return qs


def load_aime(args):
    rows = [json.loads(l) for l in open(args.data or AIME_JSONL) if l.strip()]
    return [
        {"id": str(r["problem_idx"]), "gold": int(r["answer"]), "prompt": r["problem"].strip() + "\n\n" + AIME_INSTR}
        for r in rows
    ]


TASKS = {
    "gpqa": (load_gpqa, g.extract_answer, True),
    "mmlupro": (load_mmlupro, extract_letter_mmlu, True),
    "aime": (load_aime, extract_aime, False),  # third: allow fallback to reasoning text on truncated output
}


def stable_seed(qid, offset=0):
    return g.stable_seed(str(qid), offset)


async def run_one(client, model, q, extract, args, seed_offset):
    extra = {"top_k": args.top_k}
    if args.no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    last_err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q["prompt"]}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                presence_penalty=args.presence_penalty,
                seed=stable_seed(q["id"], seed_offset),
                extra_body=extra,
            )
            ch = resp.choices[0]
            msg = ch.message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            ext = extract(content)
            if ext is None and reasoning and not (args.task == "aime" and ch.finish_reason == "length"):
                ext = extract((reasoning + "\n" + content))
            u = resp.usage
            rt = None
            det = getattr(u, "completion_tokens_details", None) if u else None
            if det is not None:
                rt = getattr(det, "reasoning_tokens", None)
            return {
                "record_id": q["id"],
                "gold": q["gold"],
                "extracted": ext,
                "correct": ext is not None and ext == q["gold"],
                "finish_reason": ch.finish_reason,
                "completion_tokens": u.completion_tokens if u else None,
                "reasoning_tokens": rt,
                "latency_s": round(time.time() - t0, 3),
                "seed_offset": seed_offset,
                "response_text": g.trunc((reasoning + "\n</think>\n" + content) if reasoning else content),
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < 3:
                await asyncio.sleep(2**attempt * args.backoff)
    return {
        "record_id": q["id"],
        "gold": q["gold"],
        "extracted": None,
        "correct": False,
        "error": last_err,
        "finish_reason": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "latency_s": None,
        "seed_offset": seed_offset,
        "response_text": "",
    }


def summarize(path, wall):
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
        "n_finish_length": sum(r.get("finish_reason") == "length" for r in ok),
        "mean_completion_tokens": statistics.mean(toks) if toks else None,
        "max_completion_tokens": toks[-1] if toks else None,
        "sum_completion_tokens": sum(toks),
        "p50_completion_tokens": pct(0.5),
        "p95_completion_tokens": pct(0.95),
        "wall_time_s_this_run": round(wall, 1),
    }
    with open(path + ".summary.json", "w") as f:
        json.dump(s, f, indent=2)
    return s


def out_paths(args):
    if args.repeats == 1 and not args.force_suffix:
        return [args.out]
    stem = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    return [f"{stem}.r{k}.jsonl" for k in range(args.repeats)]


async def run_file(client, model, qs, extract, args, path, seed_offset):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["record_id"])
                    except Exception:
                        pass
    todo = [q for q in qs if q["id"] not in done]
    print(
        f"[{os.path.basename(path)}] {len(qs)} selected, {len(done & {q['id'] for q in qs})} already done, "
        f"{len(todo)} to run (seed_offset={seed_offset})",
        flush=True,
    )
    t0 = time.time()
    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        n = nc = 0
        total = len(todo)
        lock = asyncio.Lock()
        fout = open(path, "a")

        async def worker(q):
            nonlocal n, nc
            async with sem:
                r = await run_one(client, model, q, extract, args, seed_offset)
            async with lock:
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                n += 1
                nc += bool(r["correct"])
                if n % args.log_every == 0 or n == total:
                    print(f"[{n}/{total}] running_acc={nc / n:.3f} elapsed={time.time() - t0:.0f}s", flush=True)

        await asyncio.gather(*(worker(q) for q in todo))
        fout.close()
    s = summarize(path, time.time() - t0) if os.path.exists(path) else {}
    print(json.dumps(s, indent=2), flush=True)
    return s


async def amain(args):
    from openai import AsyncOpenAI

    loader, extract, _ = TASKS[args.task]
    qs = loader(args)
    if args.task != "mmlupro" and args.ids:
        want = {l.strip() for l in open(args.ids) if l.strip()}
        qs = [q for q in qs if q["id"] in want]
    if args.limit:
        qs = qs[: args.limit]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout, max_retries=0)
    model = args.model or (await client.models.list()).data[0].id
    print(f"task={args.task} model: {model} questions={len(qs)} repeats={args.repeats}", flush=True)
    for k, path in enumerate(out_paths(args)):
        await run_file(client, model, qs, extract, args, path, args.seed_offset + k)
    await client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="default: first from GET /v1/models")
    p.add_argument("--csv", default=None, help="gpqa csv (default: gpqa_eval.DEFAULT_CSV)")
    p.add_argument("--data", default=None, help="mmlupro/aime jsonl override")
    p.add_argument("--out", required=True, help="output jsonl (with --repeats>1: <stem>.r{k}.jsonl)")
    p.add_argument("--force-suffix", action="store_true", help="use .r0 suffix even with --repeats 1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--ids",
        default=None,
        help="file of ids, one per line (mmlupro default: 500-q stratified list; 'all' = full 12032)",
    )
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0, help="gpqa choice shuffle seed")
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--backoff", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=5)
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
