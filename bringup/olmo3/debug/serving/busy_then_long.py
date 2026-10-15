"""Burst of concurrent short requests, then sequential 8K greedy requests; report OK/BAD per request."""

import asyncio, httpx, json, requests

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
                await c.post(U, json=dict(model=M, prompt=CH, max_tokens=120, temperature=0.6, top_p=0.95), timeout=900)

        await asyncio.gather(*[one(i) for i in range(48)])


for rnd in range(3):
    asyncio.run(burst())
    res = []
    for i in range(8):
        t = requests.post(
            U, json=dict(model=M, prompt=d["ids"][: P + 8000], max_tokens=6, temperature=0), timeout=900
        ).json()["choices"][0]["text"]
        res.append("OK" if t.startswith(" /11)(x") else "BAD")
    print("after burst", rnd, res, flush=True)
