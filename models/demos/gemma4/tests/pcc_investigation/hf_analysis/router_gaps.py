# How close are the 8th- and 9th-best experts? Same book text, HF bf16, CPU, sdpa vs eager.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, gc, importlib, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
P, spec = sys.argv[1], sys.argv[2]
mod_name, cls_name = spec.split(":"); Router = getattr(importlib.import_module(mod_name), cls_name)
orig = Router.forward; rec = {}
def patched(self, h):
    out = orig(self, h)
    first = out[0].float()
    probs = first if "gemma4" in mod_name else torch.softmax(first, -1)  # Gemma returns probs; OLMoE/Qwen return logits
    rec[self._layer] = (probs.detach().reshape(-1, probs.shape[-1]), out[2].detach().reshape(-1, out[2].shape[-1]))
    return out
Router.forward = patched
tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id: ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:1023]])
def run(attn):
    global rec; rec = {}
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn).eval()
    for i, r in enumerate(x for x in m.modules() if isinstance(x, Router)): r._layer = i
    with torch.no_grad(): m(ids)
    del m; gc.collect(); return dict(rec)
A = run("sdpa"); B = run("eager")
k = A[0][1].shape[-1]
gp, gl, fgp, fgl, top8_mass = [], [], [], [], []
for L in A:
    p = A[L][0]; t = p.topk(k + 1, -1).values
    gap_p = t[:, k-1] - t[:, k]; gap_l = torch.log(t[:, k-1].clamp_min(1e-30)) - torch.log(t[:, k].clamp_min(1e-30))
    flip = (torch.sort(A[L][1], -1).values != torch.sort(B[L][1], -1).values).any(-1)
    gp.append(gap_p); gl.append(gap_l); fgp.append(gap_p[flip]); fgl.append(gap_l[flip]); top8_mass.append(t[:, :k].sum(-1))
cat = lambda xs: torch.cat(xs)
gp, gl, fgp, fgl, m8 = map(cat, (gp, gl, fgp, fgl, top8_mass))
name = P.rstrip("/").split("/")[-1]
print(f"RESULT {name}: top-{k}; prob of 8th expert median {cat([A[L][0].topk(k,-1).values[:,k-1] for L in A]).median():.4f}; "
      f"8th-9th prob gap median {gp.median():.5f}, share <0.001 {(gp<0.001).float().mean()*100:.1f}%; "
      f"score gap (log p8 - log p9) median {gl.median():.4f}, share <0.01 {(gl<0.01).float().mean()*100:.1f}%; "
      f"top-8 share of probability median {m8.median():.3f}; flips {fgp.numel()} with median prob gap {fgp.median():.5f}, score gap {fgl.median():.4f}")
