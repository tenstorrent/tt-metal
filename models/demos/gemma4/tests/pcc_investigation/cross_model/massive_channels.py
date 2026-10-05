# Does the residual stream carry a few huge "massive" channels that the next norm weights almost to zero?
# Any HF MoE (Gemma 4, Qwen3-MoE, OLMoE), bf16 forward on NTOK book tokens with BOS; per layer input:
# share of the residual's energy in the top 4 channels, before and after the input norm's weight,
# and the ratio |residual| / |residual without those channels| (how much the massive channels dilute a relative error).
# Usage: massive_channels.py <model_dir> <gemma4|qwen3_moe|olmoe>
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, json, sys
import torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

P, kind = sys.argv[1], sys.argv[2]
name = P.rstrip("/").split("/")[-1]
NTOK, T0 = 512, 32
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:NTOK]])
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
layers = m.model.language_model.layers if kind == "gemma4" else m.model.layers
X = {}
for i, l in enumerate(layers):
    l.register_forward_pre_hook(lambda mod, a, k, i=i: X.__setitem__(i, (a[0] if a else k["hidden_states"])[0, T0:].float().clone()), with_kwargs=True)
with torch.no_grad():
    m(ids, use_cache=False)
rows = []
for i, l in enumerate(layers):
    x = X[i]
    w = l.input_layernorm.weight.float()
    if kind != "gemma4" and getattr(l.input_layernorm, "weight", None) is not None:
        pass  # Qwen3/OLMoE RMSNorm: out = x / rms * weight (same form as Gemma 4)
    e = (x ** 2).mean(0)
    top = e.argsort(descending=True)[:4]
    ew = w ** 2 * e
    rest = torch.ones_like(e, dtype=torch.bool); rest[top] = False
    dilution = (x.norm(dim=-1) / x[:, rest].norm(dim=-1)).mean().item()
    rows.append(dict(layer=i, top4=top.tolist(), top4_energy_pct=round((e[top].sum() / e.sum()).item() * 100, 1),
                     top4_after_norm_pct=round((ew[top].sum() / ew.sum()).item() * 100, 1),
                     top1_abs=round(x[:, top[0]].abs().mean().item(), 2), median_abs=round(x.abs().mean(0).median().item(), 4),
                     norm_w_top1=round(w[top[0]].item(), 4), norm_w_median=round(w.median().item(), 4),
                     dilution=round(dilution, 3)))
    print("MASSIVE " + json.dumps(dict(model=name, **rows[-1])), flush=True)
json.dump(rows, open(f"{DATA}/massive_channels_{name}.json", "w"), indent=1)
print("DONE", name, flush=True)
