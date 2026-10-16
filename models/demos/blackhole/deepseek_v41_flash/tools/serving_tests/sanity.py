import concurrent.futures as cf
import json
import time
import urllib.request

U = "http://localhost:8000/v1/chat/completions"
M = "deepseek-ai/DeepSeek-V4.1-Flash"


def post(body, stream=False):
    r = urllib.request.Request(U, json.dumps(dict(model=M, **body)).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=900)


def chat(p, **kw):
    t = time.time()
    d = json.load(post(dict(messages=[{"role": "user", "content": p}], **kw)))
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], time.time() - t


print("sampled:", chat("Write one sentence about the sea.", max_tokens=40, temperature=0.6, top_k=20, top_p=0.95))
# streaming
t = time.time()
resp = post(
    dict(
        messages=[{"role": "user", "content": "Count from 1 to 12, separated by commas."}],
        max_tokens=60,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
)
chunks = []
first = None
for line in resp:
    line = line.decode().strip()
    if line.startswith("data: ") and line != "data: [DONE]":
        j = json.loads(line[6:])
        if j["choices"] and j["choices"][0]["delta"].get("content"):
            if first is None:
                first = time.time() - t
            chunks.append(j["choices"][0]["delta"]["content"])
print("stream:", repr("".join(chunks)), "chunks", len(chunks), "ttft %.2fs" % first, "total %.2fs" % (time.time() - t))
qs = [
    ("What is %d*%d? Just the number." % (a, b), str(a * b))
    for a, b in [(12, 13), (17, 23), (25, 16), (31, 9), (14, 14), (19, 21), (8, 77), (45, 11)]
]
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda q: chat(q[0], max_tokens=24, temperature=0), qs))
for (q, a), (out, n, dt) in zip(qs, res):
    print("conc:", q, "->", repr(out), "OK" if a in out else "MISMATCH", "%.1fs" % dt)
