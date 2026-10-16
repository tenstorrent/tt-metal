#!/usr/bin/env python3
"""GSM8K 8-shot shared-prefix benchmark client for the APC / chunked-prefill server tests (streaming /v1/completions, greedy).
usage: apc_client.py <out.json> [--mixed] [--rounds N]
phases: burst1 (16 concurrent distinct questions, store cold), burst2 (16 other questions, warm), burst1b (burst1 again: identical prompts), solo (4 sequential), [mixed: 8 short chats + one long prompt]"""
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SHOTS = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]
HEADER = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in SHOTS)
QS = json.load(
    open(
        "/mnt/tt-data/ssinghal/wt/tm_apc/models/demos/blackhole/deepseek_v41_flash/demo/sample_prompts/input_data_gsm8k_128_with_answers.json"
    )
)


def prompt_for(i):
    return HEADER + f"Q: {QS[i]['prompt']}\nA:"


def stream(prompt, max_tokens=256, stop=()):
    body = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    if stop:
        body["stop"] = list(stop)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    times = []
    text = ""
    usage = None
    fin = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                times.append(time.time() - t0)
                text += ch[0]["delta"]["content"]
            if ch and ch[0].get("finish_reason"):
                fin = ch[0]["finish_reason"]
    return dict(
        text=text, ttft=times[0] if times else None, total=time.time() - t0, times=times, usage=usage, finish=fin
    )


def burst(idx, label):
    out = [None] * len(idx)
    t0 = time.time()

    def go(k):
        out[k] = stream(prompt_for(idx[k]))

    ts = [threading.Thread(target=go, args=(k,)) for k in range(len(idx))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    toks = sum((o["usage"] or {}).get("completion_tokens", len(o["times"])) for o in out)
    empty = sum(1 for o in out if o["ttft"] is None)
    for o in out:
        if o["ttft"] is None:
            o["ttft"] = o["total"]
    ttf = sorted(o["ttft"] for o in out)
    q = lambda p: ttf[min(len(ttf) - 1, int(p * len(ttf)))]
    tpots = [(o["total"] - o["ttft"]) / max(1, len(o["times"]) - 1) for o in out]
    res = dict(
        label=label,
        n=len(idx),
        wall=wall,
        completion_tokens=toks,
        throughput_tok_s=toks / wall,
        ttft_p50=q(0.5),
        ttft_p99=q(0.99),
        ttft_max=ttf[-1],
        tpot_mean_ms=1000 * statistics.mean(tpots),
        prompt_tokens=(out[0]["usage"] or {}).get("prompt_tokens"),
        empty=empty,
        sample=out[0]["text"][:120],
        texts=[o["text"] for o in out],
    )
    print(json.dumps({k: v for k, v in res.items() if k != "texts"}), flush=True)
    return res


def mixed(long_words, n_short=8):
    """8 short chats stream; at t=4 s one long prompt arrives. -> ITL of the short streams, TTFT of shorts / long, long total"""
    random.seed(1)
    w = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()
    long_prompt = (
        " ".join(random.choice(w) for _ in range(long_words))
        + "\n\nThe secret code is 4817. Repeat the secret code mentioned above."
    )
    shorts = [None] * n_short
    long_res = [None]
    t0 = time.time()

    def go_short(k):
        shorts[k] = stream(prompt_for(40 + k), max_tokens=256)

    def go_long():
        time.sleep(4.0)
        long_res[0] = dict(start=time.time() - t0, **stream(long_prompt, max_tokens=24, stop=()))

    ts = [threading.Thread(target=go_short, args=(k,)) for k in range(n_short)] + [threading.Thread(target=go_long)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    L = long_res[0]
    gaps = []
    for s in shorts:
        gaps += [(b - a, a) for a, b in zip(s["times"], s["times"][1:])]
    gaps.sort(key=lambda x: x[0])
    g = [x[0] for x in gaps]
    res = dict(
        label="mixed",
        long_words=long_words,
        long_prompt_tokens=(L["usage"] or {}).get("prompt_tokens"),
        long_ttft=L["ttft"],
        long_total=L["total"],
        long_text=L["text"],
        short_ttft_max=max(s["ttft"] for s in shorts),
        itl_p50=g[len(g) // 2],
        itl_p99=g[int(0.99 * len(g))],
        itl_max=g[-1],
        short_texts=[s["text"] for s in shorts],
    )
    print(json.dumps({k: v for k, v in res.items() if "text" not in k}), flush=True)
    return res


if __name__ == "__main__":
    out = sys.argv[1]
    res = {}
    res["warm"] = [
        stream(prompt_for(100 + k), max_tokens=8) for k in range(3)
    ]  # (compiles / captures the prefill + decode traces; the 2nd / 3rd request restores a snapshot (APC on): compiles the restore programs)
    res["burst1"] = burst(list(range(0, 16)), "burst1 (16 distinct questions, shared 8-shot header)")
    res["burst2"] = burst(list(range(16, 32)), "burst2 (16 other questions)")
    res["burst1b"] = burst(list(range(0, 16)), "burst1 again (identical prompts)")
    seq = [
        stream(prompt_for(i), max_tokens=160) for i in range(0, 16)
    ]  # sequential: no batch-composition noise, every request after the first restores the shared header (APC on)
    res["seq"] = [dict(ttft=x["ttft"], total=x["total"], text=x["text"], n=len(x["times"])) for x in seq]
    print("seq ttft", [round(x["ttft"], 3) for x in seq], flush=True)
    solo = []
    for i in range(48, 52):
        s = stream(prompt_for(i))
        solo.append(dict(ttft=s["ttft"], total=s["total"], text=s["text"], n=len(s["times"])))
    res["solo"] = solo
    print("solo ttft", [round(x["ttft"], 3) for x in solo], flush=True)
    if "--mixed" in sys.argv:
        res["mixed"] = mixed(int(sys.argv[sys.argv.index("--mixed") + 1]) if sys.argv[-1] != "--mixed" else 32000)
    json.dump(res, open(out, "w"))
    print("DONE", flush=True)
