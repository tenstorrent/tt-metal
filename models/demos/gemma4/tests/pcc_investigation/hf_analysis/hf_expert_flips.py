# Do Gemma 4's expert choices change between two HF runs that differ only in attention arithmetic order?
#   run A: sdpa (records experts), run B: eager (free), run C: eager but forced to use run A's experts.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, torch
import transformers.models.gemma4.modeling_gemma4 as G
from transformers import AutoModelForCausalLM
P = f"{MODELS}/gemma-4-26B-A4B-it"
ids = torch.load(f"{DATA}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][:, :1023]
orig_forward = G.Gemma4TextRouter.forward
rec = {}      # layer -> (probs [S,128], index [S,8]) for the current run
force = None  # layer -> index [S,8] to force

def patched(self, hidden_states):
    probs, w, idx = orig_forward(self, hidden_states)
    L = self._layer
    if force is not None:
        idx = force[L]
        w = probs.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        w = w * self.per_expert_scale[idx]
    rec[L] = (probs.detach().float(), idx.detach())
    return probs, w, idx
G.Gemma4TextRouter.forward = patched

def run(attn):
    global rec
    rec = {}
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn).eval()
    for i, layer in enumerate(m.model.language_model.layers):
        layer.router._layer = i
    with torch.no_grad():
        lg = m(ids).logits[0].float()
    del m; gc.collect()
    return lg, rec

def pcc_rows(a, b):
    return torch.stack([torch.corrcoef(torch.stack((a[i], b[i])))[0, 1] for i in range(a.shape[0])])

A, recA = run("sdpa")
B, recB = run("eager")
force = {L: recA[L][1] for L in recA}
C, _ = run("eager")

S = ids.shape[1]
print("layer | positions whose 8 chosen experts differ (A vs B) | of those, median gap between 8th and 9th expert prob in A")
total_diff = 0
first_layer = None
for L in sorted(recA):
    pa, ia = recA[L]; pb, ib = recB[L]
    sa = torch.sort(ia, -1).values; sb = torch.sort(ib, -1).values
    diff = (sa != sb).any(-1)
    n = int(diff.sum()); total_diff += n
    if n and first_layer is None: first_layer = L
    top9 = pa.topk(9, -1).values
    gap = (top9[:, 7] - top9[:, 8])[diff]
    print(f"{L:5d} | {n:4d} / {S} ({n/S*100:5.1f}%) | {gap.median().item() if n else float('nan'):.5f}")
any_diff = torch.zeros(S, dtype=torch.bool)
for L in recA:
    any_diff |= (torch.sort(recA[L][1], -1).values != torch.sort(recB[L][1], -1).values).any(-1)
print(f"positions with at least one layer's experts different: {int(any_diff.sum())} / {S}; first layer with a flip: {first_layer}")
allgap = torch.cat([recA[L][0].topk(9, -1).values[:, 7] - recA[L][0].topk(9, -1).values[:, 8] for L in recA])
print(f"typical 8th-vs-9th gap over all layers/positions: median {allgap.median():.5f}; share below 0.001: {(allgap<0.001).float().mean()*100:.1f}%")
s = slice(511, 1011)
for name, X in (("A sdpa vs B eager (experts free)", B), ("A sdpa vs C eager forced to A's experts", C)):
    p = pcc_rows(A, X)
    print(f"{name}: positions 511-1010 mean PCC {p[s].mean():.5f}, <0.99 {(p[s]<0.99).sum().item()}/500, "
          f"top-1 agree {(A[s].argmax(-1)==X[s].argmax(-1)).float().mean()*100:.1f}% | all positions mean {p.mean():.5f}")
