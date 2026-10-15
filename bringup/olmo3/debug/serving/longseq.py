import json, requests, sys

d = json.load(open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json"))
P = d["prompt_len"]


def run(N, label, n=4, **kw):
    for i in range(n):
        r = requests.post(
            "http://localhost:8000/v1/completions",
            json=dict(model="allenai/Olmo-3.1-32B-Think", prompt=d["ids"][: P + N], max_tokens=40, **kw),
            timeout=1200,
        )
        try:
            print(N, label, i, repr(r.json()["choices"][0]["text"][:100]), flush=True)
        except Exception:
            print(N, label, i, r.text[:100])
            return


for N in map(int, sys.argv[1:]):
    run(N, "dev", temperature=0.6, top_p=0.95)
    run(N, "host", temperature=0.6, top_p=0.95, top_k=0)
    run(N, "greedy", 2, temperature=0)
