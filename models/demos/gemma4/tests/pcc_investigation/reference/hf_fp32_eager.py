# Hugging Face Gemma 4 entirely in fp32 (CPU): the closest available stand-in for exact arithmetic.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
from transformers import AutoModelForCausalLM
D = f"{DATA}"
book = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][:, :1023]
chat = torch.load(f"{D}/chat_ids.pt")["ids"][None]
m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/gemma-4-26B-A4B-it", dtype=torch.float32, attn_implementation="eager").eval()
print("loaded", next(m.parameters()).dtype, flush=True)
with torch.no_grad():
    torch.save(m(chat).logits[0].float(), f"{D}/chat_logits_hf_fp32_eager.pt"); print("chat done", flush=True)
    torch.save(m(book).logits[0].float().to(torch.bfloat16), f"{D}/book_logits_hf_fp32_eager.pt"); print("book done", flush=True)
