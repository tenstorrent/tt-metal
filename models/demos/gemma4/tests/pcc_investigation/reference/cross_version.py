# Transformers 5.18 Gemma 4 vs the saved Transformers 5.12.1 run, same 1023 book tokens, bf16, sdpa, CPU.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import collections, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
P = f"{MODELS}/gemma-4-26B-A4B-it"
R = f"{DATA}/gemma-4-26B-A4B-it.refpt"
ids = torch.load(R)["reference_tokens"][:, :1023]
old = torch.load(R + ".logits.pt").float()
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
with torch.no_grad():
    new = m(ids).logits[0].float()
torch.save(new.to(torch.bfloat16), R + ".logits-5.18.pt")
s = slice(511, 1011)
pcc = torch.stack([torch.corrcoef(torch.stack((old[i], new[i])))[0, 1] for i in range(1023)])
real = ids[0, 1:]
print(f"transformers {transformers.__version__}: real next word guessed {(new[:-1].argmax(-1)==real).float().mean()*100:.1f}% (5.12.1: {(old[:-1].argmax(-1)==real).float().mean()*100:.1f}%)")
print(f"5.18 vs 5.12.1 logits, positions 511-1010: mean PCC {pcc[s].mean():.4f}, <0.99 {(pcc[s]<0.99).sum().item()}/500, same top word {(old[s].argmax(-1)==new[s].argmax(-1)).float().mean()*100:.1f}%")
top = new[s].argmax(-1)
print("5.18 most common top guesses:", [(tok.decode([t]), n) for t, n in collections.Counter(top.tolist()).most_common(6)])
