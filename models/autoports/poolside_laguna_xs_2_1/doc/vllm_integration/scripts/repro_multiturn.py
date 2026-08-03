# Batched-decode CORRUPTION localizer (v4) — MULTI-TURN concurrent CHAT (the real failing condition).
# Single-turn completions were clean across temp/top_k/context. The SWE corruption was multi-turn chat
# with tools: KV grows across 100+ turns + prefix-cache block reuse, N conversations concurrent.
# Here: N concurrent conversations, each K turns, temp1/top_k20, enable_thinking. Detect duplicated
# sub-tokens ('bedbed'/'ableable') in content OR reasoning per turn; report first (conv,turn) to corrupt.
import concurrent.futures as cf
import json
import re
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
DUP = re.compile(r"(\w{3,})\1")
PATHS = [f"/testbed/astropy/modeling/separable_{i}.py" for i in range(40)]


def chat(messages, mx=600):
    body = json.dumps(
        {
            "model": "poolside/Laguna-XS-2.1",
            "messages": messages,
            "max_tokens": mx,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=280)
        m = json.load(r)["choices"][0]["message"]
        return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or "")
    except Exception as e:
        return f"<ERR {type(e).__name__}>", ""


def one_conversation(cid, turns=25):
    msgs = [
        {
            "role": "user",
            "content": "You are exploring a Python repo at /testbed. I'll ask you to echo exact file paths and reason about them. Start: echo the exact path "
            + PATHS[0]
            + " verbatim.",
        }
    ]
    hits = []
    for t in range(turns):
        c, r = chat(msgs)
        if c.startswith("<ERR"):
            return {"cid": cid, "err": c, "turn": t, "hits": hits}
        blob = c + " " + r
        d = list(dict.fromkeys(DUP.findall(blob)))
        if d:
            hits.append((t, d[:6], blob[:120]))
        # grow context: append assistant + a new observation that references a path (where dup showed)
        msgs.append({"role": "assistant", "content": c})
        p = PATHS[(t + 1) % len(PATHS)]
        msgs.append(
            {
                "role": "user",
                "content": f"Good. Now cat that file and echo the exact path {p} verbatim, then explain in 2 sentences what module {t+1} does for separability of nested CompoundModels.",
            }
        )
    return {"cid": cid, "turns": turns, "hits": hits}


N = 4
print(f"[localizer v4] {N} concurrent multi-turn chat conversations, temp1/top_k20, enable_thinking", flush=True)
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(lambda i: one_conversation(i), range(N)))
for res in results:
    if res.get("err"):
        print(
            f"  conv{res['cid']}: ERROR {res['err']} at turn {res['turn']} (hits so far {len(res['hits'])})", flush=True
        )
    else:
        print(f"  conv{res['cid']}: {res['turns']} turns, corrupt turns={len(res['hits'])}", flush=True)
        for t, d, snip in res["hits"][:3]:
            print(f"     turn{t}: {d} :: ...{snip!r}...", flush=True)
tot = sum(len(r.get("hits", [])) for r in results)
print(f"TOTAL corrupt turns across all conversations: {tot}", flush=True)
print("DONE", flush=True)
