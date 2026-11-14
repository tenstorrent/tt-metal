"""CPU: pass-key needle prompts (4 users at depths 10/40/70/90 %) of ~L tokens -> torch file {ids: [4 lists], keys, depths, L} (format of integ/needle_prompts.pt). usage: gen_needle_prompts.py L OUT.pt"""
import random
import sys

import torch
from transformers import AutoTokenizer

L, OUT = int(sys.argv[1]), sys.argv[2]
tok = AutoTokenizer.from_pretrained("/mnt/tt-data/Hy4-Preview", trust_remote_code=True)
words = "river stone cloud lamp garden window engine paper silver forest harbor bridge candle market winter violin copper meadow ladder thunder".split()
depths = [0.1, 0.4, 0.7, 0.9]
ids_all, keys = [], []
for u, dep in enumerate(depths):
    rnd = random.Random(100 + u)
    sent = lambda: " ".join(rnd.choice(words) for _ in range(9)).capitalize() + f" number {rnd.randrange(10**6)}. "
    key = rnd.randrange(10000, 99999)
    q = "\n\nWhat is the pass key? Answer with the number only."
    needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
    nq = len(tok(q, add_special_tokens=False)["input_ids"]) + len(tok(needle, add_special_tokens=False)["input_ids"])

    # filler tokenised in blocks (tokenising a megabyte of text at once is slow but fine); stop at the depth and at the end
    def filler(ntok):
        out = []
        while len(out) < ntok:
            txt = "".join(sent() for _ in range(2000))
            out += tok(txt, add_special_tokens=False)["input_ids"]
        return out[:ntok]

    n_before = int((L - nq) * dep)
    ids = [tok.bos_token_id] if tok.bos_token_id is not None and False else []
    ids += (
        filler(n_before)
        + tok(needle, add_special_tokens=False)["input_ids"]
        + filler(L - nq - n_before)
        + tok(q, add_special_tokens=False)["input_ids"]
    )
    ids_all.append(ids[:L] if len(ids) >= L else ids + [tok.pad_token_id] * (L - len(ids)))
    keys.append(str(key))
    print(u, dep, len(ids), key, flush=True)
torch.save(dict(ids=ids_all, keys=keys, depths=depths, L=L), OUT)
