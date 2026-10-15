"""Collect device-sampled doubled-word glitches and ask the host reference for the probability of the repeated token."""

import asyncio, httpx, re, math, sys

U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
PR = [
    "<|im_start|>user\nWhat is the capital of France? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nExplain how a transistor works.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]


async def gen(c, sem, p):
    async with sem:
        r = await c.post(U, json=dict(model=M, prompt=p, max_tokens=300, temperature=0.6, top_p=0.95), timeout=1800)
        return p, r.json()["choices"][0]["text"]


async def ref(c, prompt, tok):
    r = await c.post(
        U,
        json=dict(model=M, prompt=prompt, max_tokens=1, temperature=0.6, top_p=0.95, top_k=0, logprobs=20),
        timeout=600,
    )
    top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    return top.get(tok), sorted(top.items(), key=lambda kv: -kv[1])[:3]


async def main():
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(32)
        outs = await asyncio.gather(*[gen(c, sem, PR[i % 2]) for i in range(int(sys.argv[1]))])
        n = 0
        for p, t in outs:
            for m in re.finditer(r"\b([A-Za-z]+) \1\b", t):
                w = " " + m.group(1)
                prefix = p + t[: m.start()] + m.group(1)
                lp, top = await ref(c, prefix, w)
                n += 1
                print(
                    f"dup {w!r:12} host logprob of repeat = {lp if lp is None else round(lp,2)}  (prob {'<e-20' if lp is None else format(math.exp(lp),'.2e')})  host top3={[(k,round(math.exp(v),3)) for k,v in top]}  ctx=...{t[max(0,m.start()-40):m.end()+15]!r}",
                    flush=True,
                )
        print("total glitches examined:", n)


asyncio.run(main())
