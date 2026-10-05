import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import sys, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
P = f"{MODELS}/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(P)
text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of"
ids = [tok.bos_token_id] + tok.encode(text)
attn = sys.argv[1]
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn).eval()
print("transformers", transformers.__version__, "attn", m.config._attn_implementation if hasattr(m.config, "_attn_implementation") else attn, "class", type(m).__name__)
with torch.no_grad():
    lg = m(torch.tensor([ids])).logits[0].float()
hits = 0
for i in range(1, len(ids)):
    top = lg[i - 1].topk(3).indices.tolist()
    hits += top[0] == ids[i]
print(f"next-word top-1 on the sentence: {hits}/{len(ids)-1}")
print("prediction after full text:", [tok.decode([t]) for t in lg[-1].topk(5).indices.tolist()])
