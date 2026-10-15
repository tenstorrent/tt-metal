import json, requests, collections

d = json.load(open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json"))
P = d["prompt_len"]
U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"


def post(n_ctx, **kw):
    r = requests.post(U, json=dict(model=M, prompt=d["ids"][:n_ctx], max_tokens=8, temperature=0, **kw), timeout=900)
    return r.status_code, (
        r.json()["choices"][0]["text"] if r.status_code == 200 else r.json().get("error", {}).get("message", "")[:150]
    )


for n in (3000, 4000, 4096, 4097, 8000):
    print(n, post(n))
outs = [post(3900)[1] for _ in range(10)]
c = collections.Counter(outs)
print("3900-token prompt x10, identical:", c.most_common(1)[0][1], "/10")
