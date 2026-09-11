"""doc0 alone under candidate loop-breaking settings. Isolated requests, sequential."""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = sys.argv[2]
rows = [json.loads(l) for l in open(sys.argv[1])]
prompt = rows[0]["doc"]["prompt"]
VARIANTS = [
    ("greedy_rep1.05",  {"temperature": 0.0, "repetition_penalty": 1.05}),
    ("greedy_nothink",  {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}),
    ("greedy_pres0.5",  {"temperature": 0.0, "presence_penalty": 0.5}),
    ("t1_p95_rep1.05",  {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.05}),
]
if len(sys.argv) > 3:
    VARIANTS = [v for v in VARIANTS if v[0] in sys.argv[3].split(",")]
for tag, extra in VARIANTS:
    payload = {"model": "zai-org/GLM-4.7-Flash",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 16384, "seed": 42, **extra}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3000))
    except urllib.error.HTTPError as e:
        print(f"[{tag}] HTTP {e.code}: {e.read()[:300].decode()}", flush=True); continue
    json.dump(d, open(f"{OUT}/loopbreak_{tag}_doc0.json", "w"))
    c = d["choices"][0]; m = c["message"]
    content = m.get("content") or ""; reas = m.get("reasoning") or m.get("reasoning_content") or ""
    dt = time.time() - t0
    print(f"[{tag}] finish={c.get('finish_reason')} tokens={d['usage']['completion_tokens']} "
          f"content={len(content)}ch reasoning={len(reas)}ch {dt:.0f}s "
          f"({d['usage']['completion_tokens']/dt:.1f} tok/s)", flush=True)
    if content: print(f"     content head: {content[:140]!r}", flush=True)
