import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
from transformers import AutoModelForCausalLM
d = torch.load(f"{DATA}/chat_ids.pt")
m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/gemma-4-26B-A4B-it", dtype=torch.bfloat16).eval()
with torch.no_grad():
    lg = m(d["ids"][None]).logits[0].float()
torch.save(lg, f"{DATA}/chat_logits_hf.pt"); print("HF done", tuple(lg.shape))
