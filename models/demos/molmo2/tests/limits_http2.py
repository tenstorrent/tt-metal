import sys

exec(
    open("ref/limits_http.py")
    .read()
    .split("# 1. exact")[0]
    .replace("P = int(sys.argv[1]); CK = sys.argv[2]", "P=8001; CK=sys.argv[1]")
)
tail2 = tok.encode(
    "\nTask: write a very long, detailed essay (at least 1500 words) about the history of this harbour town and its keeper, based on the ledger above.",
    add_special_tokens=False,
).ids
ids = head + (body * 5000)[: 34000 - len(head) - len(tail2)] + tail2
d, e, dt = post("/v1/completions", {"model": "allenai/Molmo2-8B", "prompt": ids, "max_tokens": 2048, "temperature": 0})
if not d:
    print("FAIL", e)
    sys.exit(1)
txt = d["choices"][0]["text"]
g = tok.encode(txt, add_special_tokens=False).ids
print(
    d["usage"],
    d["choices"][0]["finish_reason"],
    f"{dt:.0f}s distinct-4gram(last256)={rep(g):.2f} words={len(txt.split())}",
)
print(repr(txt[:200]))
print(repr(txt[-200:]))
