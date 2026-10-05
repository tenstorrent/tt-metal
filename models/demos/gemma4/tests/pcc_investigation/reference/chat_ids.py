import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(f"{MODELS}/gemma-4-26B-A4B-it")
msgs = [{"role": "user", "content": "Explain in three sentences why the sky is blue, and then list three primary colors."}]
ids = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)
ids = list(ids["input_ids"])
answer = ("The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, but shorter blue "
          "wavelengths are scattered much more strongly by the gases in the atmosphere. That scattered blue light "
          "reaches our eyes from every direction. The three primary colors are red, yellow, and blue.")
full = list(ids) + t.encode(answer)
torch.save({"ids": torch.tensor(full), "prompt_len": len(ids)}, f"{DATA}/chat_ids.pt")
print("CHAT", len(ids), "prompt tokens,", len(full), "total; starts", full[:6])
