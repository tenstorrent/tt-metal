import asyncio, httpx, random, re

U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
CH = "<|im_start|>user\nExplain how a transistor works.<|im_end|>\n<|im_start|>assistant\n<think>\n"


async def req(c, **kw):
    r = await c.post(U, json=dict(model=M, **kw), timeout=1800)
    return r.json()["choices"][0]["text"]


async def main():
    async with httpx.AsyncClient() as c:
        # 1. explicit seeds: same seed -> same text, different seeds -> different text
        a = await req(c, prompt=CH, max_tokens=60, temperature=0.6, top_p=0.95, seed=1234)
        b = await req(c, prompt=CH, max_tokens=60, temperature=0.6, top_p=0.95, seed=1234)
        d = await req(c, prompt=CH, max_tokens=60, temperature=0.6, top_p=0.95, seed=999)
        print("seed repeat identical:", a == b, "| different seed differs:", a != d)
        # 2. sampled differs from greedy; greedy repeatable
        g1 = await req(c, prompt=CH, max_tokens=60, temperature=0)
        g2 = await req(c, prompt=CH, max_tokens=60, temperature=0)
        s = await req(c, prompt=CH, max_tokens=60, temperature=0.6, top_p=0.95)
        print("greedy repeatable:", g1 == g2, "| sampled differs from greedy:", s != g1)
        # 3. penalties change a repetitive prompt
        rep = "Repeat the word hello forever: hello hello hello hello hello hello hello hello hello hello hello hello"
        base = await req(c, prompt=rep, max_tokens=60, temperature=0.6, top_p=0.95)
        pen = await req(
            c, prompt=rep, max_tokens=60, temperature=0.6, top_p=0.95, frequency_penalty=2.0, presence_penalty=2.0
        )
        frac = lambda t: t.lower().count("hello") / max(1, len(t.split()))
        print(f"hello fraction: no penalty {frac(base):.2f} -> penalties {frac(pen):.2f}")
        # 4. explicit seeds across 64 concurrent requests: glitch rate + all completed
        sem = asyncio.Semaphore(32)

        async def one(i):
            async with sem:
                return await req(c, prompt=CH, max_tokens=300, temperature=0.6, top_p=0.95, seed=1000 + i)

        outs = await asyncio.gather(*[one(i) for i in range(64)])
        w = sum(len(t.split()) for t in outs)
        g = sum(len(re.findall(r"\b([A-Za-z]+) \1\b", t)) for t in outs)
        print(
            f"seeded, 64 concurrent: words={w} doubled={g} per-1k={1000*g/w:.2f}; distinct outputs={len(set(outs))}/64"
        )
        # 5. mixed random configs, 96 requests at 32 concurrency
        cfgs = lambda i: random.choice(
            [
                dict(temperature=0),
                dict(temperature=0.6, top_p=0.95),
                dict(temperature=0.8, top_p=0.9, seed=i),
                dict(temperature=0.6, top_p=0.95, frequency_penalty=0.5, presence_penalty=0.5),
                dict(temperature=0.6, top_p=0.95, top_k=0),
            ]
        )

        async def mix(i):
            async with sem:
                kw = cfgs(i)
                t = await req(c, prompt=CH, max_tokens=random.choice([20, 60, 150]), **kw)
                return len(t) > 0

        ok = await asyncio.gather(*[mix(i) for i in range(96)])
        print("mixed stress ok:", sum(ok), "/96")


asyncio.run(main())
