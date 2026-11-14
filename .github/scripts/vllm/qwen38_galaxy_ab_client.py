#!/usr/bin/env python3
"""Greedy A/B client for the Qwen3.8 BH Galaxy GDN-state legs.

Each leg runs the same requests against its own server and writes every reply,
timing and prefix-cache counter to one JSON file. The two files are compared
offline, so a leg only fails here on transport errors, empty replies or replies
with U+FFFD.
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"

# Word counts of the single prompts; the last one is near one 16k-token prefill chunk.
PROMPT_WORDS = [90, 280, 700, 1500, 2600, 4200, 7000, 11000]
LONG_PROMPT_WORDS = 48000
SESSION_TURNS = [
    "Summarize the text above in three sentences.",
    "Which part of it is the hardest to implement, and why?",
    "List two follow-up questions an engineer would ask.",
    "Give the text a one-line title.",
]


def corpus_words(root):
    # Real text: random-token prompts make greedy replies loop and inflate draft acceptance.
    files = sorted(pathlib.Path(root).rglob("*.md"))
    words = []
    for f in files:
        try:
            words.extend(f.read_text(errors="ignore").split())
        except OSError:
            continue
    return words


def text_at(words, offset, n):
    if len(words) < n:
        reps = n // max(len(words), 1) + 1
        words = words * reps
    offset %= len(words) - n + 1
    return " ".join(words[offset : offset + n])


def chat(model, messages, max_tokens, timeout=1800):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "X-User-ID": "vllm_test"}
    )
    t0 = time.perf_counter()
    ttft, text, usage, err = None, "", None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                msg = json.loads(line[5:])
                if msg.get("usage"):
                    usage = msg["usage"]
                for ch in msg.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        ttft = ttft if ttft is not None else time.perf_counter() - t0
                        text += piece
    except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
        err = repr(e)
    e2e = time.perf_counter() - t0
    out_tokens = (usage or {}).get("completion_tokens")
    tpot = (e2e - ttft) / (out_tokens - 1) if ttft is not None and out_tokens and out_tokens > 1 else None
    return {
        "text": text,
        "ttft": ttft,
        "e2e": e2e,
        "tpot": tpot,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "error": err,
    }


def metrics():
    try:
        txt = urllib.request.urlopen(METRICS_URL, timeout=60).read().decode()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    keep = (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_preemptions_total",
    )
    out = {}
    for line in txt.splitlines():
        if line.startswith(keep):
            name_labels, value = line.rsplit(" ", 1)
            out[name_labels] = float(value)
    return out


def summed(m, name):
    return sum(v for k, v in m.items() if k.startswith(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="tech_reports")
    args = ap.parse_args()

    words = corpus_words(args.corpus)
    print(f"corpus: {len(words)} words from {args.corpus}")
    prompts = [text_at(words, 7919 * (i + 1), n) for i, n in enumerate(PROMPT_WORDS)]
    ask = "Summarize the text above in five short bullet points."

    def single(i):
        return [{"role": "system", "content": prompts[i]}, {"role": "user", "content": ask}]

    res = {"model": args.model, "prompt_words": PROMPT_WORDS, "metrics_start": metrics()}
    t_all = time.perf_counter()

    # 1. One request at a time: the greedy reference of this leg.
    res["b1"] = [chat(args.model, single(i), 160) for i in range(len(prompts))]

    # 2. The same prompts 4 times over, 32 at once (8 per DP rank on average).
    with cf.ThreadPoolExecutor(32) as ex:
        res["b32"] = list(ex.map(lambda k: chat(args.model, single(k % len(prompts)), 160), range(32)))

    # 3. Second turn of every single prompt: resumes from the prefix cache.
    m0 = metrics()
    res["followup"] = []
    for i in range(len(prompts)):
        msgs = single(i) + [
            {"role": "assistant", "content": res["b1"][i]["text"]},
            {"role": "user", "content": "Give the text a one-line title."},
        ]
        res["followup"].append(chat(args.model, msgs, 48))
    m1 = metrics()
    res["followup_cache"] = {
        "queried": summed(m1, "vllm:prefix_cache_queries_total") - summed(m0, "vllm:prefix_cache_queries_total"),
        "hits": summed(m1, "vllm:prefix_cache_hits_total") - summed(m0, "vllm:prefix_cache_hits_total"),
    }

    # 4. Agentic sessions: 4 shared system prompts, 16 sessions of 4 turns, all sessions at once.
    systems = [text_at(words, 104729 * (j + 1), 2500) for j in range(4)]

    def session(s):
        msgs = [{"role": "system", "content": systems[s % 4]}]
        turns = []
        for q in SESSION_TURNS:
            msgs.append({"role": "user", "content": f"(session {s}) {q}"})
            r = chat(args.model, msgs, 120)
            turns.append(r)
            msgs.append({"role": "assistant", "content": r["text"]})
        return turns

    m0 = metrics()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(16) as ex:
        res["sessions"] = list(ex.map(session, range(16)))
    res["sessions_wall"] = time.perf_counter() - t0
    m1 = metrics()
    res["sessions_cache"] = {
        "queried": summed(m1, "vllm:prefix_cache_queries_total") - summed(m0, "vllm:prefix_cache_queries_total"),
        "hits": summed(m1, "vllm:prefix_cache_hits_total") - summed(m0, "vllm:prefix_cache_hits_total"),
    }

    # 5. Long context: about 64k tokens, then a follow-up that should resume near the end.
    long_msgs = [{"role": "system", "content": text_at(words, 3, LONG_PROMPT_WORDS)}, {"role": "user", "content": ask}]
    res["long"] = chat(args.model, long_msgs, 64)
    res["long_followup"] = chat(
        args.model,
        long_msgs
        + [
            {"role": "assistant", "content": res["long"]["text"]},
            {"role": "user", "content": "Give the text a one-line title."},
        ],
        32,
    )

    # 6. Image, then a text follow-up in the same conversation.
    img_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in 50 words."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }
    ]
    res["image"] = chat(args.model, img_msgs, 96)
    res["image_followup"] = chat(
        args.model,
        img_msgs
        + [
            {"role": "assistant", "content": res["image"]["text"]},
            {"role": "user", "content": "How many animals are in the image?"},
        ],
        32,
    )

    res["wall"] = time.perf_counter() - t_all
    res["metrics_end"] = metrics()
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))

    replies = res["b1"] + res["b32"] + res["followup"] + [r for s in res["sessions"] for r in s]
    replies += [res["long"], res["long_followup"], res["image"], res["image_followup"]]
    bad = [r for r in replies if r["error"] or not r["text"].strip() or "�" in r["text"]]
    b1_vs_b32 = sum(res["b1"][k % len(prompts)]["text"] == res["b32"][k]["text"] for k in range(32))
    ttfts = sorted(t["ttft"] for s in res["sessions"] for t in s[1:] if t["ttft"] is not None)
    print(f"replies: {len(replies)}, failed or junk: {len(bad)}")
    print(f"b32 replies equal to this leg's b1: {b1_vs_b32}/32")
    print(f"follow-up cache: {res['followup_cache']}")
    print(f"sessions: wall {res['sessions_wall']:.1f} s, cache {res['sessions_cache']}")
    if ttfts:
        print(f"session follow-up TTFT p50 {ttfts[len(ttfts) // 2]:.3f} s")
    print(f"long: ttft {res['long']['ttft']}, follow-up ttft {res['long_followup']['ttft']}")
    print(f"image: {res['image']['text'][:120]!r}")
    for r in bad[:5]:
        print(f"bad reply: error={r['error']} text={r['text'][:80]!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
