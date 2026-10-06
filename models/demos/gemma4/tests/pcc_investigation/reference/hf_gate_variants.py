# How close can ANY implementation get to the accuracy gate's reference (Hugging Face bf16, sdpa attention,
# generated/optimizer_reference/)? Re-runs Hugging Face itself on the gate's tokens with only one thing changed
# and scores each against that reference over the gate's 129 positions (127..255):
#   bf16_sdpa_repeat  - identical settings again (determinism check)
#   bf16_eager        - attention computed by the eager kernel instead of sdpa (same math, different order)
#   bf16_sdpa_1thread - one CPU thread (different reduction split inside the matmuls)
#   fp32              - the saved fp32 reference (reference/hf_gate_fp32_ref.py)
import os as _os
from pathlib import Path as _Path

DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.demos.gemma4.tests import test_optimizer_gemma4_pcc as GATE

P = f"{MODELS}/gemma-4-26B-A4B-it"
tokens = GATE.encode_text(AutoTokenizer.from_pretrained(P, local_files_only=True))[: GATE.PROMPT_TOKENS + GATE.FORCED_TOKENS]
ref = GATE._reference_logits(P, tokens).float()
s = slice(GATE.PROMPT_TOKENS - 1, len(tokens))


def score(name, logits):
    a, b = logits[s].double(), ref[s].double()
    a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
    p = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    r = ref[s].argmax(-1)
    out = dict(run=name, mean_pcc=round(p.mean().item(), 5), min_pcc=round(p.min().item(), 4), below_099=int((p < 0.99).sum()),
               top1=round((logits[s].argmax(-1) == r).float().mean().item() * 100, 2),
               top5=round((logits[s].topk(5, -1).indices == r[:, None]).any(1).float().mean().item() * 100, 2))
    print("VARIANT " + json.dumps(out), flush=True)


score("fp32", torch.load(f"{DATA}/gate_hf_fp32_logits.pt").float())
ids = torch.tensor([tokens])
for name, attn, threads in (("bf16_sdpa_repeat", "sdpa", None), ("bf16_eager", "eager", None), ("bf16_sdpa_1thread", "sdpa", 1)):
    if threads:
        torch.set_num_threads(threads)
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn, local_files_only=True).eval()
    with torch.no_grad():
        logits = m(ids).logits[0].float()
    torch.save(logits.to(torch.bfloat16), f"{DATA}/gate_hf_{name}_logits.pt")
    score(name, logits)
    del m
print("DONE", flush=True)
