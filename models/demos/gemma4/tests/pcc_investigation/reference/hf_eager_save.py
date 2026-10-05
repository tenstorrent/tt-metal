import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
from transformers import AutoModelForCausalLM
D = f"{DATA}"
ids = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][:, :1023]
m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/gemma-4-26B-A4B-it", dtype=torch.bfloat16, attn_implementation="eager").eval()
with torch.no_grad():
    torch.save(m(ids).logits[0].float().to(torch.bfloat16), f"{D}/book_logits_hf_eager.pt")
print("eager saved")
