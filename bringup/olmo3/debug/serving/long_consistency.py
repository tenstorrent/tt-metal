import json, requests, sys, collections

d = json.load(open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json"))
P = d["prompt_len"]
U = "http://localhost:8000/v1/completions"


def run(N, n, kw, label):
    outs = []
    for i in range(n):
        r = requests.post(
            U, json=dict(model="allenai/Olmo-3.1-32B-Think", prompt=d["ids"][: P + N], max_tokens=24, **kw), timeout=900
        ).json()
        try:
            outs.append(r["choices"][0]["text"])
        except Exception:
            outs.append("ERR " + str(r)[:60])
            break
    c = collections.Counter(outs)
    top, cnt = c.most_common(1)[0]
    print(
        f"{label} N={N+P}: {cnt}/{len(outs)} identical to the majority; others:",
        [o[:40] for o, k in c.items() if o != top][:6],
        flush=True,
    )


for N in map(int, sys.argv[2:]):
    run(N, int(sys.argv[1]), dict(temperature=0), "greedy")
