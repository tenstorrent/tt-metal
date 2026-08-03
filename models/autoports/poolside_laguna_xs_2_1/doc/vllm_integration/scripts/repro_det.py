# Batched-decode CORRUPTION localizer (v3).
# Tests two new axes vs v2 (which was clean at temp0): (1) HETEROGENEOUS concurrent batch (users at
# different context lengths/positions, like real serving), (2) SAMPLING mode sweep. Corruption =
# duplicated sub-tokens ('bedbed'/'ableable'). Isolates on-device sampler vs attention/KV.
import concurrent.futures as cf
import json
import re
import urllib.request

URL = "http://localhost:8000/v1/completions"
DUP = re.compile(r"(\w{3,})\1")


def ctxlines(n):
    return "Repo /testbed file index:\n" + "\n".join(
        f"/testbed/pkg/module_{i}.py defines helper {i} for the analysis subsystem" for i in range(n)
    )


# heterogeneous prompts: very different context lengths in the SAME concurrent batch
LENS = [300, 1200, 2200, 3200, 4200]  # ~5k .. ~68k tokens spread
PROMPTS = [
    ctxlines(n)
    + f"\n\nList the full path of file number 3, then write a 6-sentence explanation of the analysis subsystem for module set {n}.\n"
    for n in LENS
]


def call(prompt, temp, top_k, mx=700):
    body = {"model": "poolside/Laguna-XS-2.1", "prompt": prompt, "max_tokens": mx, "temperature": temp, "seed": 0}
    if top_k:
        body["top_k"] = top_k
    if temp > 0:
        body["top_p"] = 1.0
    b = json.dumps(body).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, b, {"Content-Type": "application/json"}), timeout=280)
        return json.load(r)["choices"][0]["text"]
    except Exception as e:
        return f"<ERR {type(e).__name__}>"


def run(label, temp, top_k, rounds=2):
    total_corrupt = 0
    total = 0
    sample = None
    for _ in range(rounds):
        with cf.ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
            outs = list(ex.map(lambda p: call(p, temp, top_k), PROMPTS))
        for o in outs:
            if o.startswith("<ERR"):
                continue
            total += 1
            if DUP.findall(o):
                total_corrupt += 1
                if sample is None:
                    sample = (list(dict.fromkeys(DUP.findall(o)))[:8], o[:160])
    print(f"[{label}] heterogeneous batch, {rounds} rounds: corrupt {total_corrupt}/{total}", flush=True)
    if sample:
        print(f"   e.g. {sample[0]} :: ...{sample[1]!r}...", flush=True)


print(f"[localizer v3] heterogeneous concurrent batch lens={LENS} (~5k..68k tok), sampling sweep", flush=True)
run("temp0/argmax", 0.0, None)
run("temp1.0/top_k=1", 1.0, 1)
run("temp1.0/top_k=20", 1.0, 20)
print("DONE", flush=True)
