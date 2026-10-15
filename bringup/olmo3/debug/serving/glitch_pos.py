"""Where (absolute position mod 64) do doubled-word glitches occur? 64 = vLLM block size / page-table growth."""

import asyncio, httpx, re, sys, collections
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("/home/ubuntu/models/olmo31-32b-think")
U = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
PR = [
    "<|im_start|>user\nWhat is the capital of France? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nExplain how a transistor works.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]


async def gen(c, sem, p):
    async with sem:
        r = await c.post(
            U, json=dict(model=M, prompt=p, max_tokens=int(sys.argv[2]), temperature=0.6, top_p=0.95), timeout=3600
        )
        return p, r.json()["choices"][0]["text"]


async def main():
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(32)
        outs = await asyncio.gather(*[gen(c, sem, PR[i % 2]) for i in range(int(sys.argv[1]))])
    mods = []
    for p, t in outs:
        plen = len(tk(p, add_special_tokens=False).input_ids)
        for m in re.finditer(r"\b([A-Za-z]+) \1\b", t):
            idx = len(
                tk(t[: m.start()] + m.group(1), add_special_tokens=False).input_ids
            )  # index of the 2nd occurrence (0-based = count so far)
            mods.append((plen + idx) % 64)
            print("glitch at generated token", idx, "abs pos", plen + idx, "mod64", (plen + idx) % 64, flush=True)
    print("n", len(mods), "mod64 histogram:", sorted(collections.Counter(mods).items()))


asyncio.run(main())
