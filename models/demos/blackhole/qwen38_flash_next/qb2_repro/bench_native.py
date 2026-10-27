#!/usr/bin/env python
"""Benchmark client for the Qwen3.8-Flash-Next native chat server (tt-metal PR #58690).

Server API used (tools/qwen38_chat_server.py):
  GET  /health                  -> {"status": "ready", ...}; 200 only after READY (before that: refused / non-200)
  POST /v1/chat/completions     -> OpenAI style; timing in response["qwen38"] and response["usage"]
Thinking is ON by default on the server; this client sends enable_thinking=false so the output is plain answer text.
The server keeps a prompt-end snapshot / committed prefix and reuses it for an exact repeat or an extending prompt, so
EVERY request here starts with a unique random nonce (first tokens of the user message) and prefix_reused is checked.

Usage:  bench_native.py --url http://HOST:8000 --suite all --out-dir DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# ----------------------------------------------------------------------------------------------------------------------
# prompt construction

_SUBJECTS = [
    "The harbor town",
    "A retired engineer",
    "The old library",
    "Our research group",
    "The mountain railway",
    "A young botanist",
    "The night market",
    "An autumn storm",
    "The city council",
    "A traveling musician",
    "The village bakery",
    "A deep-sea survey crew",
    "The observatory",
    "A long river",
    "The glass workshop",
]
_VERBS = [
    "recorded",
    "remembered",
    "repaired",
    "described",
    "measured",
    "carried",
    "questioned",
    "celebrated",
    "mapped",
    "borrowed",
    "rebuilt",
    "followed",
    "sketched",
    "compared",
    "delivered",
]
_OBJECTS = [
    "a collection of weathered letters",
    "the first draft of a new timetable",
    "seven small brass instruments",
    "an unexpected change in the tides",
    "the quiet routines of winter",
    "a ledger of forgotten debts",
    "several jars of preserved plums",
    "a detailed chart of the northern coast",
    "the rumor of a hidden spring",
    "an argument about the price of bread",
    "the slow collapse of an old bridge",
    "a stack of unsorted photographs",
    "the careful arithmetic of a harvest",
    "a lamp that never stayed lit",
    "the long shadow of the clock tower",
]
_TAILS = [
    "before the first frost arrived.",
    "while the neighbours watched from their windows.",
    "although nobody could say exactly why.",
    "and then wrote everything down in plain language.",
    "long after the festival had ended.",
    "because the old rules no longer applied.",
    "with more patience than anyone expected.",
    "as the evening trains went by.",
    "until the numbers finally agreed.",
    "in the middle of an ordinary week.",
]


def filler_text(rng: random.Random, n_sentences: int) -> str:
    out = []
    for _ in range(n_sentences):
        out.append(f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TAILS)}")
    return " ".join(out)


class PromptBuilder:
    """Builds a chat user message whose chat-template rendering has exactly N tokens."""

    def __init__(self, tokenizer_dir: str):
        from transformers import AutoTokenizer  # tokenizer only

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    def chat_len(self, content: str) -> int:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return len(ids)

    def build(
        self, target: int, seed: int, instruction: str = "Continue with a long, detailed story."
    ) -> tuple[str, str, int]:
        """Return (content, nonce, actual_len).  The nonce is the very first thing in the user message."""
        rng = random.Random(seed)
        nonce = uuid.UUID(int=rng.getrandbits(128)).hex[:16] + f"-{os.getpid() % 1000}-{time.time_ns() % 100000}"
        head = f"[request {nonce}] Read the notes below. "
        tail = f" {instruction}"
        base = self.chat_len(head + tail)
        if base > target:
            raise ValueError(f"target {target} is smaller than the fixed template overhead {base}")
        body_ids: list[int] = []
        while len(body_ids) < target - base + 64:
            body_ids += self.tok.encode(" " + filler_text(rng, 40), add_special_tokens=False)
        # tokens of (head + body + tail) are not additive at the seams: search n around the estimate
        n = target - base
        best = None
        for _ in range(40):
            body = self.tok.decode(body_ids[: max(n, 0)])
            content = head + body + tail
            got = self.chat_len(content)
            if got == target:
                best = content
                break
            n += target - got
        if best is None:  # fall back to a linear scan
            for m in range(max(0, target - base - 16), target - base + 17):
                content = head + self.tok.decode(body_ids[:m]) + tail
                if self.chat_len(content) == target:
                    best = content
                    break
        if best is None:
            raise RuntimeError(f"could not build a prompt of exactly {target} tokens")
        return best, nonce, target


# ----------------------------------------------------------------------------------------------------------------------
# request machinery


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def post_chat(client: httpx.AsyncClient, url: str, payload: dict, stream: bool, timeout: float) -> dict:
    """One request.  Returns a record: client-side wall times, the server's qwen38 block and usage, or the HTTP error."""
    rec: dict[str, Any] = {
        "started_utc": utc(),
        "payload": {k: v for k, v in payload.items() if k != "messages"},
        "ok": False,
    }
    t0 = time.perf_counter()
    try:
        if not stream:
            r = await client.post(url + "/v1/chat/completions", json=payload, timeout=timeout)
            rec["client_total_s"] = time.perf_counter() - t0
            rec["http_status"] = r.status_code
            if r.status_code != 200:
                rec["error_body"] = r.text[:500]
                rec["retry_after"] = r.headers.get("retry-after")
                return rec
            doc = r.json()
            rec.update(
                ok=True,
                usage=doc.get("usage"),
                qwen38=doc.get("qwen38"),
                text=((doc.get("choices") or [{}])[0].get("message") or {}).get("content"),
                finish_reason=((doc.get("choices") or [{}])[0]).get("finish_reason"),
            )
            return rec
        usage = qwen = finish = None
        text: list[str] = []
        t_first = t_last = None
        n_chunks = 0
        async with client.stream("POST", url + "/v1/chat/completions", json=payload, timeout=timeout) as r:
            rec["http_status"] = r.status_code
            if r.status_code != 200:
                rec["error_body"] = (await r.aread()).decode(errors="replace")[:500]
                rec["retry_after"] = r.headers.get("retry-after")
                rec["client_total_s"] = time.perf_counter() - t0
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except ValueError:
                    continue
                if "error" in ch:
                    rec["stream_error"] = ch["error"]
                usage = ch.get("usage") or usage
                qwen = ch.get("qwen38") or qwen
                for choice in ch.get("choices") or []:
                    d = choice.get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content")
                    if piece:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        n_chunks += 1
                        text.append(d.get("content") or "")
                    finish = choice.get("finish_reason") or finish
        rec["client_total_s"] = time.perf_counter() - t0
        rec.update(
            ok=rec.get("stream_error") is None and usage is not None,
            usage=usage,
            qwen38=qwen,
            text="".join(text),
            finish_reason=finish,
            client_chunks=n_chunks,
        )
        if t_first is not None:
            rec["client_ttft_s"] = t_first - t0
            rec["client_decode_s"] = t_last - t_first
    except Exception as e:  # network error, timeout: record, do not crash
        rec["client_total_s"] = time.perf_counter() - t0
        rec["exception"] = f"{type(e).__name__}: {e}"
    return rec


def derive(rec: dict) -> dict:
    """Flat metrics from a record: server-reported first, client-side alongside."""
    m: dict[str, Any] = {"ok": rec.get("ok", False), "http_status": rec.get("http_status")}
    if not rec.get("ok"):
        return m
    q = rec.get("qwen38") or {}
    u = rec.get("usage") or {}
    n = u.get("completion_tokens")
    m.update(
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=n,
        queue_wait_s=u.get("queue_wait_seconds"),
        ttft_s=q.get("ttft_seconds"),
        server_tps=q.get("tokens_per_second"),
        prefill_ms_per_token=q.get("prefill_ms_per_prompt_token"),
        prefill_s=q.get("prefill_seconds"),
        prefill_tokens=q.get("prefill_tokens"),
        prefix_reused=q.get("prefix_reused"),
        prefix_restored=q.get("prefix_restored"),
        decode_loop=q.get("decode_loop"),
        finish=q.get("finish"),
        mtp_tokens_per_pass=(q.get("mtp") or {}).get("tokens_per_pass"),
    )
    tps = q.get("tokens_per_second")
    m["decode_ms_per_token"] = 1000.0 / tps if tps else None  # server: (n-1)/decode_seconds, first token excluded
    ds = q.get("decode_seconds")
    if ds and n and n >= 2:
        m["decode_ms_per_token"] = 1000.0 * ds / (n - 1)
    # client-side
    m["client_total_s"] = rec.get("client_total_s")
    m["client_ttft_s"] = rec.get("client_ttft_s")
    cd = rec.get("client_decode_s")
    if cd is not None and n and n >= 2:
        m["client_decode_ms_per_token"] = 1000.0 * cd / (n - 1)
        m["client_tps"] = (n - 1) / cd if cd > 0 else None
    return m


class Recorder:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.f = open(out_dir / "raw.jsonl", "a", buffering=1)

    def write(self, suite: str, point: str, idx: int, rec: dict, extra: dict | None = None):
        row = {"suite": suite, "point": point, "repeat": idx, **(extra or {}), **rec, "metrics": derive(rec)}
        self.f.write(json.dumps(row, default=str) + "\n")
        return row["metrics"]


# ----------------------------------------------------------------------------------------------------------------------


async def wait_ready(client: httpx.AsyncClient, url: str, limit_s: float = 1200.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < limit_s:
        try:
            r = await client.get(url + "/health", timeout=10)
            if r.status_code == 200:
                doc = r.json()
                if doc.get("status") == "ready":
                    print(f"[ready] after {time.time() - t0:.0f}s", flush=True)
                    return doc
                last = f"status={doc.get('status')}"
            else:
                last = f"http {r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}"
        await asyncio.sleep(5)
    raise SystemExit(f"server not ready after {limit_s:.0f}s (last: {last})")


def base_payload(content: str, max_tokens: int, ignore_eos: bool, stream: bool) -> dict:
    p: dict[str, Any] = {
        "model": "Qwen/Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": stream,
    }
    if stream:
        p["stream_options"] = {"include_usage": True}
    if ignore_eos:
        p["ignore_eos"] = True
    return p


def with_greedy(p: dict) -> dict:
    return {**p, "temperature": 0}


def stats(vals: list[float]) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)}


METRICS = [
    "ttft_s",
    "client_ttft_s",
    "decode_ms_per_token",
    "client_decode_ms_per_token",
    "server_tps",
    "prefill_ms_per_token",
    "prefill_s",
    "queue_wait_s",
    "completion_tokens",
    "prefix_reused",
    "client_total_s",
]


def summarize(points: dict[str, list[dict]]) -> dict:
    out = {}
    for name, ms in points.items():
        okm = [m for m in ms if m.get("ok")]
        s = {
            "requests": len(ms),
            "ok": len(okm),
            "failed": len(ms) - len(okm),
            "http_503": sum(1 for m in ms if m.get("http_status") == 503),
        }
        for k in METRICS:
            st = stats([m.get(k) for m in okm])
            if st:
                s[k] = st
        out[name] = s
    return out


# ----------------------------------------------------------------------------------------------------------------------
# suites


async def suite_decode_like(args, client, pb, rec, name: str, sampled: bool, points: dict):
    point = name
    for i in range(args.repeats):
        content, nonce, _ = pb.build(128, seed=args.seed + 1000 * (1 + sampled) + i)
        p = base_payload(content, args.max_tokens or 256, not args.no_ignore_eos, not args.no_stream)
        if sampled:
            p.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed + 1 + i)
        else:
            p = with_greedy(p)
        r = await post_chat(client, args.url, p, not args.no_stream, args.timeout)
        m = rec.write(name, point, i, r, {"nonce": nonce, "target_prompt_tokens": 128})
        points.setdefault(point, []).append(m)
        _log(name, point, i, m)


async def suite_prefill(args, client, pb, rec, points: dict):
    for isl in [int(x) for x in args.isl_list.split(",")]:
        for i in range(args.repeats):
            content, nonce, _ = pb.build(isl, seed=args.seed + 7 * isl + i)
            p = with_greedy(base_payload(content, args.max_tokens or 16, False, not args.no_stream))
            r = await post_chat(client, args.url, p, not args.no_stream, max(args.timeout, 3600))
            m = rec.write("prefill", f"isl{isl}", i, r, {"nonce": nonce, "target_prompt_tokens": isl})
            if m.get("ok") and (m.get("prefix_reused") or m.get("prompt_tokens") != isl):
                m["warning"] = f"prefix_reused={m.get('prefix_reused')} prompt_tokens={m.get('prompt_tokens')} != {isl}"
                print("  WARNING:", m["warning"], flush=True)
            points.setdefault(f"isl{isl}", []).append(m)
            _log("prefill", f"isl{isl}", i, m)


async def suite_concurrency(args, client, pb, rec, points: dict, agg: dict):
    for c in [int(x) for x in args.concurrency_list.split(",")]:
        for i in range(args.repeats):
            payloads = []
            for j in range(c):
                content, nonce, _ = pb.build(128, seed=args.seed + 100000 + 1000 * c + 10 * i + j)
                payloads.append(
                    (
                        with_greedy(
                            base_payload(content, args.max_tokens or 256, not args.no_ignore_eos, not args.no_stream)
                        ),
                        nonce,
                    )
                )
            t0 = time.perf_counter()
            res = await asyncio.gather(
                *[post_chat(client, args.url, p, not args.no_stream, args.timeout * 4) for p, _ in payloads]
            )
            wall = time.perf_counter() - t0
            ms = []
            tot = 0
            for j, ((_, nonce), r) in enumerate(zip(payloads, res)):
                m = rec.write(
                    "concurrency", f"c{c}", i, r, {"nonce": nonce, "request_index": j, "concurrency": c, "wall_s": wall}
                )
                ms.append(m)
                if m.get("ok"):
                    tot += m.get("completion_tokens") or 0
            points.setdefault(f"c{c}", []).extend(ms)
            a = {
                "concurrency": c,
                "repeat": i,
                "wall_s": wall,
                "output_tokens": tot,
                "aggregate_tok_s": tot / wall if wall > 0 else None,
                "ok": sum(1 for m in ms if m.get("ok")),
                "http_503": sum(1 for m in ms if m.get("http_status") == 503),
            }
            agg.setdefault(f"c{c}", []).append(a)
            rec.f.write(json.dumps({"suite": "concurrency", "point": f"c{c}", "aggregate": a}) + "\n")
            print(
                f"  [c{c} r{i}] wall {wall:.1f}s tokens {tot} agg {a['aggregate_tok_s'] or 0:.1f} tok/s "
                f"ok {a['ok']}/{c} 503s {a['http_503']}",
                flush=True,
            )


def _log(suite, point, i, m):
    if not m.get("ok"):
        print(f"  [{suite} {point} r{i}] FAILED http={m.get('http_status')}", flush=True)
        return
    f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
    print(
        f"  [{suite} {point} r{i}] ttft {f(m.get('ttft_s'), '.3f')}s (client {f(m.get('client_ttft_s'), '.3f')}) "
        f"decode {f(m.get('decode_ms_per_token'), '.2f')} ms/tok prefill {f(m.get('prefill_ms_per_token'), '.3f')} "
        f"ms/tok n={m.get('completion_tokens')} reused={m.get('prefix_reused')}",
        flush=True,
    )


def write_md(path: Path, summary: dict, aggs: dict, health: dict):
    L = [
        "# Qwen3.8-Flash-Next native server benchmark",
        "",
        f"url: {summary['meta']['url']}  ",
        f"date: {summary['meta']['date']}  ",
        f"server fingerprint: {health.get('system_fingerprint')}  ",
        f"lanes: {health.get('lanes')}  ",
        "",
    ]

    def row(name, s):
        g = lambda k, f=".3f": ("-" if k not in s else f"{s[k]['median']:{f}} [{s[k]['min']:{f}}, {s[k]['max']:{f}}]")
        return (
            f"| {name} | {s['ok']}/{s['requests']} | {s['http_503']} | {g('ttft_s')} | {g('client_ttft_s')} | "
            f"{g('decode_ms_per_token', '.2f')} | {g('server_tps', '.2f')} | {g('prefill_ms_per_token')} | "
            f"{g('completion_tokens', '.0f')} | {g('prefix_reused', '.0f')} |"
        )

    hdr = (
        "| point | ok | 503 | TTFT s (server) | TTFT s (client) | decode ms/tok | tok/s | prefill ms/tok | "
        "out tokens | prefix_reused |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    for suite, pts in summary["suites"].items():
        L += [f"## {suite}  (median [min, max])", "", hdr]
        L += [row(n, s) for n, s in pts.items()]
        L.append("")
        if suite in aggs:
            L += [
                "| concurrency | repeat | wall s | output tokens | aggregate tok/s | ok | 503 |",
                "|---|---|---|---|---|---|---|",
            ]
            for key, lst in aggs[suite].items():
                for a in lst:
                    L.append(
                        f"| {a['concurrency']} | {a['repeat']} | {a['wall_s']:.2f} | {a['output_tokens']} | "
                        f"{(a['aggregate_tok_s'] or 0):.2f} | {a['ok']} | {a['http_503']} |"
                    )
            L.append("")
    path.write_text("\n".join(L))


async def main_async(args):
    out = Path(args.out_dir)
    rec = Recorder(out)
    suites = ["decode", "sampled", "prefill", "concurrency"] if args.suite == "all" else [args.suite]
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        health = await wait_ready(client, args.url, 1200)
        print(
            "[health]",
            json.dumps(
                {
                    k: health.get(k)
                    for k in ("model", "context_limit", "sampling", "lanes", "limits", "sampling_defaults", "mtp")
                },
                default=str,
            )[:1500],
            flush=True,
        )
        (out / "health.json").write_text(json.dumps(health, indent=1, default=str))
        pb = PromptBuilder(args.tokenizer_dir)
        # one warmup request, not recorded
        content, _, _ = pb.build(128, seed=args.seed - 1)
        w = await post_chat(client, args.url, with_greedy(base_payload(content, 32, False, False)), False, 1800)
        print(
            f"[warmup] ok={w.get('ok')} http={w.get('http_status')} ttft={(w.get('qwen38') or {}).get('ttft_seconds')}",
            flush=True,
        )
        summary: dict[str, Any] = {"meta": {"url": args.url, "date": utc(), "args": vars(args)}, "suites": {}}
        aggs: dict[str, dict] = {}
        for s in suites:
            print(f"== suite {s}", flush=True)
            pts: dict[str, list[dict]] = {}
            if s == "decode":
                await suite_decode_like(args, client, pb, rec, "decode", False, pts)
            elif s == "sampled":
                # the sampled loop is a different decode path: one unrecorded sampled warmup
                content, _, _ = pb.build(128, seed=args.seed - 2)
                await post_chat(
                    client,
                    args.url,
                    {
                        **base_payload(content, 32, False, False),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "seed": 1,
                    },
                    False,
                    1800,
                )
                await suite_decode_like(args, client, pb, rec, "sampled", True, pts)
            elif s == "prefill":
                await suite_prefill(args, client, pb, rec, pts)
            elif s == "concurrency":
                aggs["concurrency"] = {}
                await suite_concurrency(args, client, pb, rec, pts, aggs["concurrency"])
            summary["suites"][s] = summarize(pts)
            if s in aggs:
                summary["suites"][s]["_aggregate"] = {
                    k: {
                        "aggregate_tok_s": stats([a["aggregate_tok_s"] for a in v]),
                        "wall_s": stats([a["wall_s"] for a in v]),
                    }
                    for k, v in aggs[s].items()
                }
            (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        md_summary = {
            **summary,
            "suites": {k: {n: p for n, p in v.items() if not n.startswith("_")} for k, v in summary["suites"].items()},
        }
        write_md(out / "summary.md", md_summary, aggs, health)
        print(f"[done] {out}/raw.jsonl summary.json summary.md", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL (server default port 8000)")
    ap.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("HF_MODEL"),
        help="flat checkpoint dir holding the tokenizer files (default: $HF_MODEL)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suite", choices=["decode", "sampled", "prefill", "concurrency", "all"], default="all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--isl-list", default="1024,4096,8192,16384,31000")
    ap.add_argument("--concurrency-list", default="1,2,4")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override max_tokens (defaults: decode/sampled/concurrency 256, prefill 16)",
    )
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--seed", type=int, default=1234, help="base seed (sampling seeds and prompt nonces derive from it)"
    )
    ap.add_argument(
        "--no-ignore-eos", action="store_true", help="let the model stop at EOS (default: force full max_tokens)"
    )
    ap.add_argument("--no-stream", action="store_true", help="non-streaming requests (no client-side TTFT)")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request HTTP timeout, seconds")
    args = ap.parse_args()
    if not args.tokenizer_dir:
        ap.error("--tokenizer-dir or HF_MODEL is required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
