"""burst of concurrent short requests, then sequential 8K / 16K greedy requests; OK/BAD per request"""

import asyncio, httpx, json, requests, collections

d = json.load(open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json"))
P = d["prompt_len"]
U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
CH = "<|im_start|>user\nExplain how a transistor works.<|im_end|>\n<|im_start|>assistant\n<think>\n"


async def burst():
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(32)

        async def one(i):
            async with sem:
                await c.post(U, json=dict(model=M, prompt=CH, max_tokens=100, temperature=0.6, top_p=0.95), timeout=900)

        await asyncio.gather(*[one(i) for i in range(40)])


def ask(n):
    r = requests.post(U, json=dict(model=M, prompt=d["ids"][: P + n], max_tokens=6, temperature=0), timeout=900).json()
    return r["choices"][0]["text"] if "choices" in r else "ERR"


ref = {}
for n in (8000, 16000):
    ref[n] = collections.Counter([ask(n) for _ in range(3)]).most_common(1)[0][0]
    print("reference", n, repr(ref[n]), flush=True)
for rnd in range(3):
    asyncio.run(burst())
    for n in (8000, 16000):
        res = ["OK" if ask(n) == ref[n] else "BAD" for _ in range(5)]
        print(f"after burst {rnd}: {n+P} tokens", res, flush=True)
