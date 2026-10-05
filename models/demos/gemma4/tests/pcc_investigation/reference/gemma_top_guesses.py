import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, random, collections, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
P = f"{MODELS}/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
words = text.split()[:2000]; random.Random(0).shuffle(words)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
for name, t in (("book", text), ("shuffled", " ".join(words))):
    ids = [tok.bos_token_id] + tok.encode(t)
    ids = torch.tensor([ids[:1023]])
    with torch.no_grad():
        p = torch.softmax(m(ids).logits[0, 511:1011].float(), -1)
    top = p.argmax(-1); conf = p.max(-1).values
    c = collections.Counter(top.tolist())
    print(f"== {name}: 10 most common top guesses over 500 positions (token, count, mean confidence)")
    for t_id, n in c.most_common(10):
        print(f"   {tok.decode([t_id])!r:22} {n:4d}  {conf[top == t_id].mean():.2f}")
