# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy-output checks of a running Qwen3.6 / Qwen3.8 vLLM server (stdlib only), for QWEN36_MAMBA_SPEC A/B runs.

    python mamba_spec_server_check.py run --url http://localhost:8000 --out greedy.json
    python mamba_spec_server_check.py prefix --url http://localhost:8000 --out prefix.json --block 2048
    python mamba_spec_server_check.py compare a.json b.json

``run``: the prompts (slices of the long sample article, 200 to 3000 words) one at a time, then all at once, greedy
(temperature 0), and records the completions. ``prefix``: a prompt of a few blocks, then the same prompt again (a
prefix-cache hit on the last block boundary), then a prompt that shares only its first blocks; each output is
recorded with the server's prefix-cache hit counter before and after. ``compare``: equal completions by key.
"""

import argparse
import json
import pathlib
import re
import sys
import threading
import time
import urllib.request

ARTICLE = pathlib.Path(__file__).resolve().parents[1] / "demo" / "sample_prompts" / "input_data_long_4k.json"


def _post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _metric(url, name):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return None
    vals = [float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M)]
    return sum(vals) if vals else None


def _model(url):
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=60) as r:
        return json.loads(r.read())["data"][0]["id"]


def _complete(url, model, prompt, max_tokens):
    t0 = time.time()
    out = _post(
        url.rstrip("/") + "/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True},
    )
    return {"text": out["choices"][0]["text"], "usage": out.get("usage"), "seconds": round(time.time() - t0, 3)}


def _words():
    return json.loads(ARTICLE.read_text())[0]["prompt"].split()


def _prompts(n):
    words = _words()
    sizes = [200, 450, 700, 1000, 1400, 1800, 2400, 3000][:n]
    out = {}
    for i, s in enumerate(sizes):
        start = (97 * i) % max(1, len(words) - s)
        out[f"p{i}_{s}w"] = " ".join(words[start : start + s]) + "\n\nSummarize the text above in detail."
    return out


def cmd_run(a):
    model = a.model or _model(a.url)
    prompts = _prompts(a.n)
    res = {"sequential": {}, "concurrent": {}}
    for k, p in prompts.items():
        res["sequential"][k] = _complete(a.url, model, p, a.max_tokens)
        print(f"sequential {k}: {res['sequential'][k]['seconds']} s", flush=True)
    threads = []

    def one(k, p):
        res["concurrent"][k] = _complete(a.url, model, p, a.max_tokens)

    t0 = time.time()
    for k, p in prompts.items():
        threads.append(threading.Thread(target=one, args=(k, p)))
        threads[-1].start()
    for t in threads:
        t.join()
    res["concurrent_seconds"] = round(time.time() - t0, 3)
    res["spec"] = {
        n: _metric(a.url, n)
        for n in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total")
    }
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}; concurrent {res['concurrent_seconds']} s; spec counters {res['spec']}")


def _stream(url, model, prompt, max_tokens):
    """A streamed greedy completion: seconds to the first token, and from the first token to the last (decode)."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}
    body["stream"] = True
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    stamps = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                stamps.append(time.time())
    return {"ttft": stamps[0] - t0, "decode": stamps[-1] - stamps[0], "stamps": stamps}


def cmd_perf(a):
    """Decode speed: ``n`` short prompts at once, ``max_tokens`` each, greedy, streamed. Step time = a request's decode
    time (first to last token, mean over requests) / its verify steps, steps = (max_tokens - 1) / tokens per step
    (vLLM's spec counters)."""
    model = a.model or _model(a.url)
    words = _words()
    prompts = [" ".join(words[97 * i : 97 * i + a.words]) + "\n\nContinue the text." for i in range(a.n)]
    names = (
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
    )
    m0 = {n: _metric(a.url, n) for n in names}
    res = [None] * a.n

    def one(i):
        res[i] = _stream(a.url, model, prompts[i], a.max_tokens)

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    d = {n: (None if m0[n] is None else _metric(a.url, n) - m0[n]) for n in names}
    acc, drafts = d[names[0]], d[names[2]]
    per_step = None if not drafts else 1 + acc / drafts  # tokens per verify step (accepted drafts + the bonus)
    dec = sum(r["decode"] for r in res) / a.n
    # one streamed chunk per verify step: the median chunk interval while every request is decoding is the step time
    lo, hi = max(r["stamps"][0] for r in res), min(r["stamps"][-1] for r in res)
    gaps = sorted(y - x for r in res for x, y in zip(r["stamps"], r["stamps"][1:]) if x >= lo and y <= hi)
    out = {
        "step_ms_median_chunk_gap": None if not gaps else round(1000 * gaps[len(gaps) // 2], 2),
        "chunk_gaps": len(gaps),
        "n": a.n,
        "max_tokens": a.max_tokens,
        "wall_s": round(wall, 3),
        "tokens_per_step": per_step,
        "draft_acceptance": None if not d[names[1]] else acc / d[names[1]],
        "decode_tok_per_s_per_user": round((a.max_tokens - 1) / dec, 2),
        "step_ms": None if not per_step else round(1000 * dec * per_step / (a.max_tokens - 1), 2),
        "ttft_s_max": round(max(r["ttft"] for r in res), 3),
        "spec_counters": d,
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "spec_counters"}))


def cmd_ttft(a):
    """Cold prefill latency: one request at a time, 2 tokens each, ``reps`` + 1 per prompt length in words (each a
    different slice of the article, so no prefix hit); the first of each length is dropped (one-time costs). Prints the
    median and every time to the first streamed token, in ms."""
    model = a.model or _model(a.url)
    words = _words()
    out = {}
    for n in (int(x) for x in a.words.split(",")):
        times = []
        for r in range(a.reps + 1):
            pool = words * (n // len(words) + 2)
            start = (53 * r) % len(words)
            prompt = " ".join(pool[start : start + n]) + "\n\nContinue the text."
            times.append(1000 * _stream(a.url, model, prompt, 2)["ttft"])
        times = times[1:]
        out[str(n)] = {"median_ms": round(sorted(times)[len(times) // 2], 1), "ms": [round(x, 1) for x in times]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def _ids(url, model, ids, max_tokens):
    out = _post(
        url.rstrip("/") + "/v1/completions",
        {
            "model": model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    return out["choices"][0]["token_ids"], out["choices"][0]["text"]


def cmd_cross(a):
    """Block boundaries under batching: ``n`` concurrent greedy requests whose prompts end 3, 10, 17, ... tokens before
    the block boundary ``block`` (each a different slice of the article), ``max_tokens`` each, so every row decodes
    across the boundary at its own step. Writes the outputs in the ``run`` file format (part "concurrent")."""
    model = a.model or _model(a.url)
    toks = _post(a.url.rstrip("/") + "/tokenize", {"model": model, "prompt": " ".join(_words() * 2)})["tokens"]
    res = {"concurrent": {}}

    def one(i):
        L = a.block - 3 - 7 * i
        ids, text = _ids(a.url, model, toks[37 * i : 37 * i + L], a.max_tokens)
        res["concurrent"][str(i)] = {"prompt_tokens": L, "ids": ids, "text": text}

    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}: {len(res['concurrent'])} requests")


def cmd_followup(a):
    """Decode-written boundary states: prompts of ``lens`` tokens decode across the block boundary ``block``; a
    follow-up = prompt + its output up to ``block + 40`` tokens + a question then hits the boundary state decode left
    (needs a retention that keeps decode boundaries). Records prefix-cache hit tokens and the follow-up outputs."""
    model = a.model or _model(a.url)
    words = " ".join(_words())
    toks = _post(a.url.rstrip("/") + "/tokenize", {"model": model, "prompt": words})["tokens"]
    q = _post(a.url.rstrip("/") + "/tokenize", {"model": model, "prompt": "\n\nWhat is this text about?"})["tokens"]
    res = {}
    for L in [int(x) for x in a.lens.split(",")]:
        first = toks[:L]
        out, _ = _ids(a.url, model, first, a.block + 64 - L)
        f = first + out[: a.block + 40 - L] + q
        h0 = _metric(a.url, "vllm:prefix_cache_hits_total")
        fo, ft = _ids(a.url, model, f, a.max_tokens)
        hit = None if h0 is None else _metric(a.url, "vllm:prefix_cache_hits_total") - h0
        res[str(L)] = {"followup_ids": f, "first_out": out, "out": fo, "text": ft, "prefix_hits": hit}
        print(f"L={L}: follow-up {len(f)} tokens, prefix hit tokens {hit}", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(res))


def cmd_replay(a):
    """Send the follow-ups of a ``followup`` file to this server (cold, or a reference build); compare outputs."""
    model = a.model or _model(a.url)
    ref = json.loads(pathlib.Path(a.src).read_text())
    same = 0
    for L, r in ref.items():
        fo, _ = _ids(a.url, model, r["followup_ids"], len(r["out"]))
        eq = fo == r["out"]
        same += eq
        n = next((i for i, (x, y) in enumerate(zip(fo, r["out"])) if x != y), None)
        print(f"L={L}: {'equal' if eq else f'first different token {n}'}")
    print(f"{same} of {len(ref)} follow-ups equal")


def cmd_prefix(a):
    model = a.model or _model(a.url)
    words = _words()
    # about 1.3 tokens per word: base spans a little over `blocks` blocks, so its repeat hits blocks * block tokens
    n = int(a.blocks * a.block / 1.3) + 400
    base = " ".join(words[:n]) if n <= len(words) else " ".join((words * (n // len(words) + 1))[:n])
    other = " ".join(base.split()[: int(a.block / 1.3)]) + " " + " ".join(reversed(words[:600]))
    res = {}
    for key, p in (
        ("cold", base + "\n\nSummarize."),
        ("hit", base + "\n\nSummarize."),
        ("shared", other + "\n\nSummarize."),
    ):
        q0 = _metric(a.url, "vllm:prefix_cache_queries_total")
        h0 = _metric(a.url, "vllm:prefix_cache_hits_total")
        r = _complete(a.url, model, p, a.max_tokens)
        r["prefix_queries"] = None if q0 is None else _metric(a.url, "vllm:prefix_cache_queries_total") - q0
        r["prefix_hits"] = None if h0 is None else _metric(a.url, "vllm:prefix_cache_hits_total") - h0
        res[key] = r
        print(f"{key}: prompt {r['usage']}, hit tokens {r['prefix_hits']}, {r['seconds']} s", flush=True)
    res["hit_equals_cold"] = res["hit"]["text"] == res["cold"]["text"]
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}; hit output == cold output: {res['hit_equals_cold']}")


def cmd_compare(a):
    x, y = (json.loads(pathlib.Path(p).read_text()) for p in (a.a, a.b))
    same = diff = 0
    for part in ("sequential", "concurrent", "cold", "hit", "shared"):
        if part not in x or part not in y:
            continue
        xs, ys = x[part], y[part]
        items = xs.items() if "text" not in xs else [(part, xs)]
        for k, v in items:
            w = ys.get(k) if "text" not in ys else ys
            if w is None:
                continue
            if v["text"] == w["text"]:
                same += 1
            else:
                diff += 1
                n = next(
                    (i for i, (c, d) in enumerate(zip(v["text"], w["text"])) if c != d),
                    min(len(v["text"]), len(w["text"])),
                )
                print(f"DIFF {part}/{k}: first difference at character {n}")
    print(f"{same} equal, {diff} different")
    return 1 if diff else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--model", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=128)
    p = sub.add_parser("prefix")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=96)
    f = sub.add_parser("perf")
    f.add_argument("--url", default="http://localhost:8000")
    f.add_argument("--model", default=None)
    f.add_argument("--out", required=True)
    f.add_argument("--n", type=int, default=8)
    f.add_argument("--words", type=int, default=300)
    f.add_argument("--max-tokens", type=int, default=512)
    x = sub.add_parser("cross")
    x.add_argument("--url", default="http://localhost:8000")
    x.add_argument("--model", default=None)
    x.add_argument("--out", required=True)
    x.add_argument("--block", type=int, default=2048)
    x.add_argument("--n", type=int, default=8)
    x.add_argument("--max-tokens", type=int, default=160)
    t = sub.add_parser("ttft")
    t.add_argument("--url", default="http://localhost:8000")
    t.add_argument("--model", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--words", default="300,1500,3000")
    t.add_argument("--reps", type=int, default=5)
    u = sub.add_parser("followup")
    u.add_argument("--url", default="http://localhost:8000")
    u.add_argument("--model", default=None)
    u.add_argument("--out", required=True)
    u.add_argument("--block", type=int, default=2048)
    u.add_argument("--lens", default="2040,2043,2045,2047")
    u.add_argument("--max-tokens", type=int, default=64)
    r2 = sub.add_parser("replay")
    r2.add_argument("--url", default="http://localhost:8000")
    r2.add_argument("--model", default=None)
    r2.add_argument("src")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    cmds = {"run": cmd_run, "perf": cmd_perf, "prefix": cmd_prefix, "followup": cmd_followup, "replay": cmd_replay}
    cmds["ttft"] = cmd_ttft
    cmds["cross"] = cmd_cross
    return {**cmds, "compare": cmd_compare}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
