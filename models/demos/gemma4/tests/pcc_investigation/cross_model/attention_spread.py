# Why are Gemma 4's attention probabilities so sensitive? Softmax probabilities move by p_j * (ds_j - sum_k p_k ds_k)
# when the scores s move by ds, so a small relative change in queries/keys (ds proportional to s) moves the probabilities
# in proportion to how spread out the scores are where the probability mass sits: the standard deviation of the
# scores under the attention distribution, sd_p(s) = sqrt(sum_j p_j (s_j - sum_k p_k s_k)^2).
# Any HF MoE (Gemma 4, Qwen3-MoE, OLMoE); plain bf16 HF run, statistics computed in fp32, eager attention, NTOK book tokens with BOS.
# Per layer (mean over heads and positions T0..): sd_p(s), entropy of p, largest p, and the attention scale used.
# Usage: attention_spread.py <model_dir> <gemma4|qwen3_moe|olmoe>
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, json, sys
import torch
import torch.nn as nn
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

P, kind = sys.argv[1], sys.argv[2]
name = P.rstrip("/").split("/")[-1]
NTOK, T0 = 512, 32
if kind == "gemma4":
    import transformers.models.gemma4.modeling_gemma4 as M
elif kind == "qwen3_moe":
    import transformers.models.qwen3_moe.modeling_qwen3_moe as M
else:
    import transformers.models.olmoe.modeling_olmoe as M

STATS = {}
_eager = M.eager_attention_forward
def eager_fwd(module, query, key, value, attention_mask, *a, **k):
    out, w = _eager(module, query, key, value, attention_mask, *a, **k)
    scaling = k.get("scaling", a[0] if a else None)
    if scaling is None:
        scaling = module.head_dim ** -0.5
    s = torch.matmul(query.float(), M.repeat_kv(key, module.num_key_value_groups).float().transpose(2, 3)) * scaling  # [1, H, S, S]
    p = w.float()[0, :, T0:]          # [H, S-T0, S]
    s = s.float()[0, :, T0:]
    mask = p > 0
    s = torch.where(mask, s, torch.zeros_like(s))
    mean = (p * s).sum(-1, keepdim=True)
    sd = ((p * (s - mean) ** 2).sum(-1)).sqrt()                   # spread of the scores under p
    ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    STATS[module.layer_idx] = dict(sd_p=round(sd.mean().item(), 3), entropy=round(ent.mean().item(), 3),
                                   max_p=round(p.max(-1).values.mean().item(), 3), scaling=round(float(scaling), 5),
                                   q_rms=round(query.float().pow(2).mean().sqrt().item(), 3),
                                   k_rms=round(key.float().pow(2).mean().sqrt().item(), 3), head_dim=query.shape[-1])
    return out, w


M.eager_attention_forward = eager_fwd
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:NTOK]])
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").eval()

with torch.no_grad():
    m(ids, use_cache=False)
rows = [dict(layer=i, **STATS[i]) for i in sorted(STATS)]
for r in rows:
    print("SPREAD " + json.dumps(dict(model=name, **r)), flush=True)
json.dump(rows, open(f"{DATA}/attention_spread_{name}.json", "w"), indent=1)
print("DONE", name, flush=True)
