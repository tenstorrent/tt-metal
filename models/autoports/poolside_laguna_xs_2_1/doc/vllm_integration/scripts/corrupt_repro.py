import concurrent.futures as cf
import json
import re
import urllib.request

URL = "http://localhost:8000/v1/completions"
dup_re = re.compile(r"(\w{3,})\1")
ctx = "Repo /testbed index:\n" + "\n".join(
    f"/testbed/astropy/modeling/separable_module_{i}.py defines CompoundModel transform helper {i} for separability analysis of nested models"
    for i in range(1200)
)  # ~8k tokens
prompt = (
    ctx
    + "\n\nEcho the exact full path of file number 3, 77, 400, and 900 above verbatim, then explain separability_matrix for nested CompoundModels in 4 sentences.\n"
)


def call(temp):
    body = json.dumps(
        {"model": "poolside/Laguna-XS-2.1", "prompt": prompt, "max_tokens": 500, "temperature": temp}
    ).encode()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=280)
    return json.load(r)["choices"][0]["text"]


def rep(lbl, o):
    a = dup_re.findall(o)
    print(f"  [{lbl}] dup_artifacts={len(a)} ex={list(dict.fromkeys(a))[:10]}", flush=True)


print(f"prompt ~{len(prompt)//4} tokens", flush=True)
print("SOLO temp1:", flush=True)
rep("solo", call(1.0))
print("5 CONCURRENT temp1 (mimics workers=5 long-context):", flush=True)
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    outs = list(ex.map(lambda i: call(1.0), range(5)))
for i, o in enumerate(outs):
    rep(f"conc{i}", o)
print("DONE", flush=True)
