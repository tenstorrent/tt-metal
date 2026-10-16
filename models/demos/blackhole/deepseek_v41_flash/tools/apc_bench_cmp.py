import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("A =", sys.argv[1], " B =", sys.argv[2])
for k in ("burst1", "burst2", "burst1b"):
    ta, tb = a[k]["texts"], b[k]["texts"]
    same = sum(x == y for x, y in zip(ta, tb))
    print(f"{k}: identical texts {same}/{len(ta)}; empty A {a[k].get('empty')} B {b[k].get('empty')}")
    for f in (
        "wall",
        "throughput_tok_s",
        "ttft_p50",
        "ttft_p99",
        "ttft_max",
        "tpot_mean_ms",
        "completion_tokens",
        "prompt_tokens",
    ):
        print(f"   {f:18s} A {a[k][f]:10.3f}  B {b[k][f]:10.3f}")
if "seq" in a and "seq" in b:
    print(
        "seq identical",
        sum(x["text"] == y["text"] for x, y in zip(a["seq"], b["seq"])),
        "/",
        len(a["seq"]),
        " mean ttft A",
        round(sum(x["ttft"] for x in a["seq"]) / len(a["seq"]), 3),
        "B",
        round(sum(x["ttft"] for x in b["seq"]) / len(b["seq"]), 3),
    )
sa, sb = a["solo"], b["solo"]
print(
    "solo identical",
    sum(x["text"] == y["text"] for x, y in zip(sa, sb)),
    "/",
    len(sa),
    " ttft A",
    [round(x["ttft"], 3) for x in sa],
    "B",
    [round(x["ttft"], 3) for x in sb],
)
if "mixed" in a and "mixed" in b:
    for f in ("long_prompt_tokens", "long_ttft", "long_total", "short_ttft_max", "itl_p50", "itl_p99", "itl_max"):
        print(f"mixed {f:18s} A {a['mixed'][f]}  B {b['mixed'][f]}")
    print(
        "mixed long text same:",
        a["mixed"]["long_text"] == b["mixed"]["long_text"],
        "| shorts identical",
        sum(x == y for x, y in zip(a["mixed"]["short_texts"], b["mixed"]["short_texts"])),
        "/",
        len(a["mixed"]["short_texts"]),
    )
