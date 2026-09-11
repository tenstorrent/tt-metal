"""Greedy doc0 alone vs greedy doc0 batched with doc1, doc2. Dumps full messages."""
import json, sys, hashlib, urllib.request, concurrent.futures as cf, time
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = sys.argv[2]
rows = [json.loads(l) for l in open(sys.argv[1])]
prompts = [r["doc"]["prompt"] for r in rows]

def post(i, tag):
    payload = {"model": "zai-org/GLM-4.7-Flash",
               "messages": [{"role": "user", "content": prompts[i]}],
               "max_tokens": 16384, "temperature": 0.0, "seed": 42}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3000))
    c = d["choices"][0]
    json.dump(d, open(f"{OUT}/greedy_{tag}_doc{i}.json", "w"))
    content = c["message"].get("content") or ""
    reas = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
    return (i, c.get("finish_reason"), d["usage"]["completion_tokens"], len(content), len(reas),
            hashlib.sha256(content.encode()).hexdigest()[:12],
            hashlib.sha256(reas.encode()).hexdigest()[:12], round(time.time()-t0))

print("=== greedy doc0 ALONE ===", flush=True)
r = post(0, "alone")
print(f"  doc0 finish={r[1]} tokens={r[2]} content={r[3]}ch reasoning={r[4]}ch "
      f"content_sha={r[5]} reasoning_sha={r[6]} {r[7]}s", flush=True)
print("=== greedy doc0+doc1+doc2 BATCHED ===", flush=True)
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    res = sorted(ex.map(lambda i: post(i, "batched"), range(3)))
for i, fin, tok, lc, lr, hc, hr, s in res:
    print(f"  doc{i} finish={fin} tokens={tok} content={lc}ch reasoning={lr}ch "
          f"content_sha={hc} reasoning_sha={hr} {s}s", flush=True)
a = json.load(open(f"{OUT}/greedy_alone_doc0.json"))["choices"][0]["message"]
b = json.load(open(f"{OUT}/greedy_batched_doc0.json"))["choices"][0]["message"]
print("message keys:", sorted(a.keys()))
print("doc0 content identical alone vs batched:", (a.get("content") or "") == (b.get("content") or ""))
ra = a.get("reasoning_content") or a.get("reasoning") or ""
rb = b.get("reasoning_content") or b.get("reasoning") or ""
print("doc0 reasoning identical alone vs batched:", ra == rb, f"(len {len(ra)} vs {len(rb)})")
