"""Second round: production sampling mode with the two remaining config candidates."""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = sys.argv[2]
rows = [json.loads(l) for l in open(sys.argv[1])]
prompt = rows[0]["doc"]["prompt"]
VARIANTS = [
    ("t1_p95_nothink", {"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}}),
    ("t1_p95_topk20",  {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
    ("t0.7_p1.0",      {"temperature": 0.7, "top_p": 1.0}),
]
for tag, extra in VARIANTS:
    payload = {"model": "zai-org/GLM-4.7-Flash", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 16384, "seed": 42, **extra}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3000))
    json.dump(d, open(f"{OUT}/loopbreak2_{tag}_doc0.json", "w"))
    c = d["choices"][0]; m = c["message"]
    content = m.get("content") or ""; reas = m.get("reasoning") or ""
    dt = time.time() - t0
    print(f"[{tag}] finish={c.get('finish_reason')} tokens={d['usage']['completion_tokens']} content={len(content)}ch "
          f"reasoning={len(reas)}ch {dt:.0f}s ({d['usage']['completion_tokens']/dt:.1f} tok/s)", flush=True)
    if content: print(f"     content head: {content[:140]!r}", flush=True)
