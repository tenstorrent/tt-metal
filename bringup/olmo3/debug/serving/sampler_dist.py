"""Compare device-sampled token frequencies with the distribution implied by the host's raw logprobs."""

import asyncio, httpx, math, collections, sys

URL = "http://localhost:8000/v1/completions"
M = "allenai/Olmo-3.1-32B-Think"
T, TOP_P = 0.6, 0.95
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "My favorite animal at the zoo is the"


async def post(c, **kw):
    r = await c.post(URL, json=dict(model=M, **kw), timeout=1200)
    return r.json()["choices"][0]


def expected(top):  # top: {token: raw logprob}, from the host full-vocab logprobs
    items = sorted(top.items(), key=lambda kv: -kv[1])[:32]
    w = [(t, math.exp(lp / T)) for t, lp in items]
    s = sum(x for _, x in w)
    p = [(t, x / s) for t, x in w]
    out = []
    c = 0
    for t, x in p:  # top-p
        out.append((t, x))
        c += x
        if c >= TOP_P:
            break
    s = sum(x for _, x in out)
    return {t: x / s for t, x in out}


def report(name, obs, exp):
    n = sum(obs.values())
    tv = 0.5 * sum(abs(obs.get(t, 0) / n - exp.get(t, 0)) for t in set(obs) | set(exp))
    outside = sum(v for t, v in obs.items() if t not in exp) / n
    print(f"== {name}: n={n} TV-distance={tv:.3f} fraction-outside-expected-support={outside:.3f}")
    for t, p in sorted(exp.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {t!r:14} expected {p:.3f} observed {obs.get(t, 0) / n:.3f}")
    bad = sorted(((v, t) for t, v in obs.items() if t not in exp), reverse=True)[:8]
    print("   outside:", [(t, v) for v, t in bad])


async def main():
    async with httpx.AsyncClient() as c:
        h = await post(c, prompt=PROMPT, max_tokens=1, temperature=0.6, top_p=0.95, top_k=0, logprobs=20)
        top = h["logprobs"]["top_logprobs"][0]
        exp1 = expected(top)
        mode = max(exp1, key=exp1.get)
        print("host top-20 sum prob:", round(sum(math.exp(v) for v in top.values()), 3), "mode", repr(mode))
        sem = asyncio.Semaphore(32)

        async def one(**kw):
            async with sem:
                return (await post(c, **kw))["text"]

        outs = await asyncio.gather(*[one(prompt=PROMPT, max_tokens=2, temperature=0.6, top_p=0.95) for _ in range(N)])
        # first token: split on the tokenizer is awkward; compare the first character run
        first = collections.Counter()
        second = collections.Counter()
        for t in outs:
            first[t] = first.get(t, 0) + 1
        # use 1-token requests for the clean first-token histogram (prefill sampling)
        o1 = await asyncio.gather(*[one(prompt=PROMPT, max_tokens=1, temperature=0.6, top_p=0.95) for _ in range(N)])
        report("prefill-sampled first token (device)", collections.Counter(o1), exp1)
        # decode step: condition on the first token == mode, compare the second token to the host distribution of PROMPT+mode
        h2 = await post(c, prompt=PROMPT + mode, max_tokens=1, temperature=0.6, top_p=0.95, top_k=0, logprobs=20)
        exp2 = expected(h2["logprobs"]["top_logprobs"][0])
        sec = collections.Counter(t[len(mode) :] for t in outs if t.startswith(mode) and len(t) > len(mode))
        # token boundary is ambiguous with strings; report string-level only for tokens that appear in the expected set
        report("decode-sampled second token | first==mode (device)", sec, exp2)


asyncio.run(main())
