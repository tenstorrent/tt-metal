import asyncio, httpx, re, sys

U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
PR = [
    "<|im_start|>user\nWhat is the capital of France? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nExplain how a transistor works.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]
N = int(sys.argv[1])
TOK = int(sys.argv[2])


async def one(c, sem, p, **kw):
    async with sem:
        r = await c.post(U, json=dict(model=M, prompt=p, max_tokens=TOK, **kw), timeout=1800)
        return r.json()["choices"][0]["text"]


def glitch(t):
    return len(re.findall(r"\b([A-Za-z]+) \1\b", t))


async def main():
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(32)
        for name, kw in (
            ("device T0.6 p0.95 k32", dict(temperature=0.6, top_p=0.95)),
            ("host   T0.6 p0.95 full", dict(temperature=0.6, top_p=0.95, top_k=0)),
        ):
            outs = await asyncio.gather(*[one(c, sem, PR[i % 2], **kw) for i in range(N)])
            words = sum(len(t.split()) for t in outs)
            g = sum(glitch(t) for t in outs)
            print(
                f"{name}: samples={N} words={words} doubled-word glitches={g} per-1k-words={1000*g/words:.2f}",
                flush=True,
            )
            if "device" in name:
                dev = outs
            else:
                host = outs
        for lab, o in (("dev", dev), ("host", host)):
            print(lab, "example:", repr(o[3][:300]))


asyncio.run(main())
