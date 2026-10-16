import json
import random
import threading
import time
import urllib.request

U = "http://localhost:8000/v1/chat/completions"
M = "deepseek-ai/DeepSeek-V4.1-Flash"


def chat(p, mt=24, to=3600):
    body = dict(model=M, messages=[{"role": "user", "content": p}], max_tokens=mt, temperature=0)
    t = time.time()
    d = json.load(
        urllib.request.urlopen(
            urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=to
        )
    )
    return d["usage"], repr(d["choices"][0]["message"]["content"])[:60], time.time() - t


print("short:", chat("What is 17*23? Answer with just the number.")[1:])
print("short:", chat("What is the capital of France? One word.")[1:])
rows = [json.loads(l)["prompt"] for l in open(__import__("os").environ.get("GSM8K_PROMPTS", "gsm8k_8shot_256.jsonl"))]
for rnd in range(2):
    res = [None] * 4

    def w(i):
        res[i] = chat(rows[rnd * 4 + i], mt=256)

    t0 = time.time()
    th = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [x.start() for x in th]
    [x.join() for x in th]
    wall = time.time() - t0
    toks = [r[0]["completion_tokens"] for r in res]
    print("gsm8k answers", [r[1][:70] for r in res])
    print(
        "gsm8k x4 round %d: wall %.1fs, tokens %s, per-user tok/s %s"
        % (rnd, wall, toks, [round(n / r[2], 1) for n, r in zip(toks, res)])
    )
random.seed(1)
w_ = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()
text = (
    " ".join(random.choice(w_) for _ in range(100000))
    + "\n\nThe secret code is 4817. Repeat the secret code mentioned above."
)
text2 = (
    "The secret code is 4817. Remember it.\n\n"
    + " ".join(random.choice(w_) for _ in range(100000))
    + "\n\nWhat is the secret code mentioned at the very start? Answer with just the number."
)
for nm, tx in (("needle-end", text), ("needle-start", text2)):
    u, c, t = chat(tx, mt=24)
    print(nm, u["prompt_tokens"], "tokens, answer", c, "time %.1fs" % t)
print("2+2?", chat("2+2?")[1:])
