# Expert-flip check for any HF MoE whose router class returns (logits_or_probs, weights, indices).
# Two HF runs on CPU, bf16, same weights and tokens; only attn_implementation differs (sdpa vs eager).
# Usage: hf_expert_flips_any.py <model_dir> <module.path:RouterClass> [gemma_fp32_router]
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, gc, importlib, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P, spec = sys.argv[1], sys.argv[2]
gemma_fp32 = "gemma_fp32_router" in sys.argv[3:]
shuffled = "shuffled" in sys.argv[3:]
mod_name, cls_name = spec.split(":")
mod = importlib.import_module(mod_name)
Router = getattr(mod, cls_name)
orig = Router.forward

if gemma_fp32:
    # Gemma 4 router with scores, softmax and top-k in fp32 (HF computes them in bf16).
    def gemma_router_fp32(self, hidden_states):
        h = self.norm(hidden_states)
        h = h.float() * self.scale.float() * self.scalar_root_size
        scores = torch.nn.functional.linear(h, self.proj.weight.float())
        probs = torch.softmax(scores, dim=-1)
        w, idx = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        w = (w * self.per_expert_scale.float()[idx]).to(hidden_states.dtype)
        return probs, w, idx
    orig = gemma_router_fp32

rec = {}
def patched(self, hidden_states):
    out = orig(self, hidden_states)
    rec[self._layer] = out[2].detach().reshape(-1, out[2].shape[-1])
    return out
Router.forward = patched

tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
if shuffled:
    import random
    words = text.split()[:2000]
    random.Random(0).shuffle(words)  # same words, random order: next word unpredictable
    text = " ".join(words)
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:1023]])

def run(attn):
    global rec
    rec = {}
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation=attn).eval()
    routers = [x for x in m.modules() if isinstance(x, Router)]
    for i, r in enumerate(routers):
        r._layer = i
    with torch.no_grad():
        lg = m(ids).logits[0].float()
    del m; gc.collect()
    return lg, dict(rec), len(routers)

A, ra, n_layers = run("sdpa")
B, rb, _ = run("eager")
S = ids.shape[1]
flip_rates = []
any_flip = torch.zeros(S, dtype=torch.bool)
for L in sorted(ra):
    d = (torch.sort(ra[L], -1).values != torch.sort(rb[L], -1).values).any(-1)
    flip_rates.append(d.float().mean().item() * 100)
    any_flip |= d
pcc = torch.stack([torch.corrcoef(torch.stack((A[i], B[i])))[0, 1] for i in range(S)])
s = slice(511, 1011)
name = P.rstrip("/").split("/")[-1] + (" [fp32 router]" if gemma_fp32 else "") + (" [shuffled words]" if shuffled else " [book]")
k = ra[0].shape[-1]
print(f"RESULT {name}: {n_layers} MoE layers, top-{k}; experts differ (sdpa vs eager) per layer: "
      f"first {flip_rates[0]:.1f}%, middle {flip_rates[len(flip_rates)//2]:.1f}%, last {flip_rates[-1]:.1f}%, "
      f"mean {sum(flip_rates)/len(flip_rates):.1f}%; positions with any flip {int(any_flip.sum())}/{S}")
print(f"RESULT {name}: positions 511-1010 mean PCC {pcc[s].mean():.4f}, <0.99 {(pcc[s]<0.99).sum().item()}/500, "
      f"top-1 agree {(A[s].argmax(-1)==B[s].argmax(-1)).float().mean()*100:.1f}%; "
      f"HF predicts real next token {(A[:-1].argmax(-1)==ids[0,1:]).float().mean()*100:.1f}%")
pr = torch.softmax(A[s], -1); ent = -(pr * torch.log(pr.clamp_min(1e-30))).sum(-1)
print(f"RESULT {name}: mean entropy of next-word guesses (nats, higher = less sure) {ent.mean():.2f}; mean top-1 probability {pr.max(-1).values.mean():.3f}")
