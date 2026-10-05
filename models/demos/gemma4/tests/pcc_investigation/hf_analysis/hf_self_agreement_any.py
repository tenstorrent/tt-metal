# Same check as hf_self_agreement.py for any model: HF sdpa vs HF eager, bf16, CPU,
# first 1024 tokens of tale-of-two-cities.txt.bz2 with the start token, as generate_reference_hf.py encodes it.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, gc, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
P = sys.argv[1]
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:1023]])
out = {}
for attn in ("sdpa", "eager"):
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn).eval()
    with torch.no_grad():
        out[attn] = m(ids).logits[0].float()
    del m; gc.collect()
a, b = out["sdpa"], out["eager"]
pcc = torch.stack([torch.corrcoef(torch.stack((a[i], b[i])))[0, 1] for i in range(a.shape[0])])
real = ids[0, 1:]
acc = (a[:-1].argmax(-1) == real).float().mean() * 100
s = slice(511, 1011)
print(f"RESULT {P.rstrip('/').split('/')[-1]}: positions 511-1010: mean PCC {pcc[s].mean():.4f}, min {pcc[s].min():.4f}, "
      f"<0.99 {(pcc[s]<0.99).sum().item()}/500, top-1 agree {(a[s].argmax(-1)==b[s].argmax(-1)).float().mean()*100:.1f}% | "
      f"all positions mean PCC {pcc.mean():.4f} | HF predicts real next token {acc:.1f}%")
