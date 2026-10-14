import json
import sys
import time
import urllib.request

from tokenizers import Tokenizer

P = int(sys.argv[1])
CK = sys.argv[2]
tok = Tokenizer.from_file(f"{CK}/tokenizer.json")


def post(path, body, t=1800):
    r = urllib.request.Request(
        f"http://localhost:{P}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=t))
    except Exception as e:
        return None, f"{e} {getattr(e,'read',lambda:b'')()[:300]}", time.time() - t0
    return d, None, time.time() - t0


def rep(ids, w=256, n=4):
    ids = ids[-w:]
    g = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / max(len(g), 1)


para = (
    "The harbour town of Eskerby keeps a ledger of every ship that has ever passed its lighthouse. "
    "The keeper, Marit, records the name, the cargo and the weather, and she has done so for forty years. "
)
head = tok.encode("Read the following ledger and answer the question at the end.\n", add_special_tokens=False).ids
tail = tok.encode("\nQuestion: Who is the keeper of the lighthouse? Answer in one word.", add_special_tokens=False).ids
body = tok.encode(para, add_special_tokens=False).ids


def prompt(total):
    return head + (body * (total // len(body) + 1))[: total - len(head) - len(tail)] + tail


# 1. exact limit: 36848 prompt + 16 new = 36864
ids = prompt(36848)
assert len(ids) == 36848
d, e, dt = post("/v1/completions", {"model": "allenai/Molmo2-8B", "prompt": ids, "max_tokens": 16, "temperature": 0})
print(
    "EXACT LIMIT 36848+16:",
    e or (d["usage"], d["choices"][0]["finish_reason"], repr(d["choices"][0]["text"])),
    f"{dt:.0f}s",
    flush=True,
)
# 2. one over the limit must be rejected cleanly, and the server must stay alive
d, e, dt = post(
    "/v1/completions", {"model": "allenai/Molmo2-8B", "prompt": prompt(36848), "max_tokens": 17, "temperature": 0}
)
print("OVER LIMIT 36848+17 :", (e or "ACCEPTED?!")[:200], flush=True)
# 3. long generation, card max_new_tokens=2048, chat, ignore_eos to force full length
d, e, dt = post(
    "/v1/chat/completions",
    {
        "model": "allenai/Molmo2-8B",
        "max_tokens": 2048,
        "temperature": 0,
        "ignore_eos": True,
        "messages": [
            {"role": "user", "content": "Write a long, detailed story about a lighthouse keeper. Keep going."}
        ],
    },
)
if d:
    txt = d["choices"][0]["message"]["content"]
    ids2 = tok.encode(txt, add_special_tokens=False).ids
    print(
        "LONG GEN 2048 (ignore_eos):",
        d["usage"],
        d["choices"][0]["finish_reason"],
        f"{dt:.0f}s distinct-4gram(last256)={rep(ids2):.2f}",
        repr(txt[-200:]),
        flush=True,
    )
else:
    print("LONG GEN FAIL", e)
# 4. long gen at high S: ~30k-token prompt + 2048 new
d, e, dt = post(
    "/v1/completions",
    {"model": "allenai/Molmo2-8B", "prompt": prompt(34000), "max_tokens": 2048, "temperature": 0, "ignore_eos": True},
)
if d:
    txt = d["choices"][0]["text"]
    ids3 = tok.encode(txt, add_special_tokens=False).ids
    print(
        "LONG GEN 2048 @S=34000:",
        d["usage"],
        d["choices"][0]["finish_reason"],
        f"{dt:.0f}s distinct-4gram={rep(ids3):.2f}",
        repr(txt[:120]),
        flush=True,
    )
else:
    print("LONG GEN@34k FAIL", e)
print("health:", urllib.request.urlopen(f"http://localhost:{P}/health").status)
