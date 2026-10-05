# HF vs HF: same weights, same bf16, same 1024 tokens; only the attention implementation (order of arithmetic) differs.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
from transformers import AutoModelForCausalLM
P = f"{MODELS}/gemma-4-26B-A4B-it"
ref = torch.load(f"{DATA}/gemma-4-26B-A4B-it.refpt")
ids = ref["reference_tokens"][:, :1023]
base = torch.load(f"{DATA}/gemma-4-26B-A4B-it.refpt.logits.pt").float()  # default (sdpa) run
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").eval()
with torch.no_grad():
    other = m(ids).logits[0].float()
pcc = torch.stack([torch.corrcoef(torch.stack((base[i], other[i])))[0, 1] for i in range(1023)])
top1 = (base.argmax(-1) == other.argmax(-1)).float().mean() * 100
s = slice(511, 1011)  # the 500 positions the chip test scores
print(f"HF sdpa vs HF eager, all 1023 positions: mean PCC {pcc.mean():.4f}, min {pcc.min():.4f}, <0.99: {(pcc<0.99).sum().item()}, top-1 agree {top1:.1f}%")
print(f"same 500 positions as chip test: mean PCC {pcc[s].mean():.4f}, min {pcc[s].min():.4f}, <0.99: {(pcc[s]<0.99).sum().item()}, top-1 agree {(base[s].argmax(-1)==other[s].argmax(-1)).float().mean()*100:.1f}%")
