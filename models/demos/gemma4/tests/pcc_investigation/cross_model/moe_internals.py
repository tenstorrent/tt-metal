# Side-by-side internals of a HF MoE model in bf16 on the book text (1023 tokens, CPU, eager attention):
#   residual stream size per layer, update-to-residual ratio per layer, attention score size per layer.
# Usage: moe_internals.py <model_dir> <transformers.models.X.modeling_X>
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, importlib, json, sys
import torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

P, modpath = sys.argv[1], sys.argv[2]
M = importlib.import_module(modpath)
name = P.rstrip("/").split("/")[-1]
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:1023]])

attn_stats = []  # per attention call: (scaling, mean over query rows of the row's largest |score|, 99th pct |score|)
orig = M.eager_attention_forward


def patched(module, query, key, value, attention_mask, *a, scaling=None, **kw):
    sc = scaling if scaling is not None else module.head_dim ** -0.5
    k = M.repeat_kv(key, module.num_key_value_groups)
    s = torch.matmul(query.float(), k.float().transpose(2, 3)) * sc  # [1, heads, T, T]
    if attention_mask is not None:
        valid = attention_mask[..., : s.shape[-1]] > -1  # unmasked positions
        valid = valid.expand_as(s)
    else:
        valid = torch.ones_like(s, dtype=torch.bool)
    absq = s.abs().masked_fill(~valid, 0)
    rowmax = absq.amax(-1)  # largest |score| per query row
    flat = s[valid].abs()
    attn_stats.append(dict(scaling=float(sc), head_dim=int(query.shape[-1]),
                           mean_row_max_abs=float(rowmax[..., 16:].mean()),
                           p99_abs=float(torch.quantile(flat[torch.randperm(flat.numel())[:200000]], 0.99))))
    return orig(module, query, key, value, attention_mask, *a, scaling=scaling, **kw)


M.eager_attention_forward = patched
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").eval()
with torch.no_grad():
    out = m(ids, output_hidden_states=True)
hs = [h[0, 16:].float() for h in out.hidden_states]  # residual stream entering each layer (skip first 16 positions)
rows = []
for i in range(len(hs) - 1):
    res, nxt = hs[i], hs[i + 1]
    rms = res.pow(2).mean(-1).sqrt().mean().item()
    upd = (nxt - res).pow(2).mean(-1).sqrt().mean().item()
    rows.append(dict(layer=i, residual_rms=round(rms, 3), residual_max_abs=round(res.abs().max().item(), 1),
                     update_rms=round(upd, 3), update_to_residual=round(upd / rms, 4)))
for i, a in enumerate(attn_stats):
    if i < len(rows):
        rows[i].update(attn_scaling=round(a["scaling"], 4), attn_head_dim=a["head_dim"],
                       attn_row_max_abs_score=round(a["mean_row_max_abs"], 2), attn_p99_abs_score=round(a["p99_abs"], 2))
json.dump(rows, open(f"{DATA}/internals-{name}.json", "w"), indent=1)
med = lambda k: sorted(r[k] for r in rows if k in r)[len(rows) // 2]
print(f"INTERNALS {name}: layers {len(rows)}; median over layers: residual RMS {med('residual_rms')}, residual max |x| {med('residual_max_abs')}, "
      f"update/residual {med('update_to_residual')}, attention scaling {rows[0].get('attn_scaling')}, "
      f"row-max |score| {med('attn_row_max_abs_score')}, p99 |score| {med('attn_p99_abs_score')}", flush=True)
for r in rows:
    print("LAYER", json.dumps(r), flush=True)
