"""Passphrase recall through the server: cold prompts 1K-32K, then a repeat (full prefix hit) and a
sibling sharing the first 75% (partial hit). Exit 1 on any miss."""
import random, sys, requests

U = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/completions"
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa".split()


def prompt(n, pp, seed, nonce, tail_seed=None):
    rng = random.Random(seed)
    w = [rng.choice(WORDS) for _ in range(n)]
    if tail_seed is not None:
        cut = int(n * 0.75)
        rt = random.Random(tail_seed)
        w[cut:] = [rt.choice(WORDS) for _ in range(n - cut)]
    at = int(n * 0.5)
    w[at:at] = f"The secret passphrase is {pp}.".split()
    return (
        f"{nonce}Read the text and answer.\n\n"
        + " ".join(w)
        + "\n\nQuestion: what is the secret passphrase?\nAnswer: the secret passphrase is"
    )


def ask(p):
    r = requests.post(
        U, json={"model": "llama70b", "prompt": p, "max_tokens": 16, "temperature": 0}, timeout=600
    ).json()
    c = r["choices"][0]["text"]
    d = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    return c, r["usage"]["prompt_tokens"], d


miss = 0
for words in [600, 1700, 2300, 3000, 4500, 9000, 18000]:
    pp = f"cobalt-heron-{words}"
    base = prompt(words, pp, seed=words, nonce=f"N{words}.\n")
    sib = prompt(words, pp, seed=words, nonce=f"N{words}.\n", tail_seed=words + 1)
    for kind, p in (("cold", base), ("repeat", base), ("sibling", sib)):
        out, ntok, cached = ask(p)
        ok = pp in out
        miss += not ok
        print(f"{kind:8s} tokens={ntok:6d} cached={cached} ok={ok} out={out[:50]!r}", flush=True)
print(f"MISSES {miss}")
sys.exit(1 if miss else 0)
