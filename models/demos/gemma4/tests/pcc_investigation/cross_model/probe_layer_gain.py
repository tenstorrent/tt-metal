# Where inside one decoder layer does a small difference grow? Same probe on any HF MoE (Gemma 4, Qwen3-MoE, OLMoE).
# Weights stored bf16, all arithmetic fp32 (upcast at use), eager attention, NTOK book tokens with BOS.
#  1. clean full forward: capture every layer's exact input (and its call arguments) and the experts it picked
#  2. full forward with a 2^-9 relative nudge on the embedding output, routing frozen to the clean run's experts:
#     capture every layer's input -> the real direction the difference takes inside the model at that layer
#  3. per layer, run that layer alone on: clean input; clean + nudge of relative size EPS in (a) a random direction,
#     (b) the real direction. Routing frozen to the clean experts, and again with routing free.
#     For every internal point (each submodule's output, plus the residual after attention), report
#     gain = (relative change at that point) / (relative change of the layer input).
# Usage: probe_layer_gain.py <model_dir> <gemma4|qwen3_moe|olmoe>
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, json, sys, time
import torch
import torch.nn as nn
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

P, kind = sys.argv[1], sys.argv[2]
name = P.rstrip("/").split("/")[-1]
NTOK, EPS, T0 = 512, 2.0 ** -9, 32  # positions T0.. are scored (skips BOS and the first few tokens)
if kind == "gemma4":
    import transformers.models.gemma4.modeling_gemma4 as M
    Router, Experts = M.Gemma4TextRouter, M.Gemma4TextExperts
elif kind == "qwen3_moe":
    import transformers.models.qwen3_moe.modeling_qwen3_moe as M
    Router, Experts = M.Qwen3MoeTopKRouter, M.Qwen3MoeExperts
else:
    import transformers.models.olmoe.modeling_olmoe as M
    Router, Experts = M.OlmoeTopKRouter, M.OlmoeExperts

FREEZE = {}   # layer -> expert indices to force
PICKED = {}   # layer -> expert indices picked in the last call
NOISE = {"on": False}


def router_fwd(self, hidden_states):
    if kind == "gemma4":
        h = self.norm(hidden_states.float())
        h = h * self.scale.float() * self.scalar_root_size
        k = self.config.top_k_experts
        logits = nn.functional.linear(h, self.proj.weight.float())
    else:
        h = hidden_states.reshape(-1, self.hidden_dim).float()
        k = self.top_k
        logits = nn.functional.linear(h, self.weight.float())
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    w, idx = torch.topk(probs, k, dim=-1)
    if self._layer in FREEZE:
        idx = FREEZE[self._layer]
        w = probs.gather(-1, idx)
    PICKED[self._layer] = idx.detach()
    if kind == "gemma4":
        w = w / w.sum(-1, keepdim=True)
        w = w * self.per_expert_scale.float()[idx]
        return probs, w, idx
    if self.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    return logits, w, idx


def experts_fwd(self, hidden_states, top_k_index, top_k_weights):
    hidden_states = hidden_states.float()
    final = torch.zeros_like(hidden_states)
    mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e[0]
        pos, tok = torch.where(mask[e])
        gate, up = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e].float()).chunk(2, dim=-1)
        h = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e].float())
        final.index_add_(0, tok, h * top_k_weights[tok, pos, None].float())
    return final


Router.forward = router_fwd
Experts.forward = experts_fwd

tok = AutoTokenizer.from_pretrained(P)
text = bz2.open(f"{REPO}/models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=True)
if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
    ids = [tok.bos_token_id] + ids
ids = torch.tensor([ids[:NTOK]])

m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").eval()


def _lin(self, x):
    return nn.functional.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


for mod in m.modules():
    if isinstance(mod, nn.Linear):
        mod.forward = _lin.__get__(mod)
emb = m.get_input_embeddings()
scale = float(getattr(emb, "scalar_embed_scale", 1.0))


def _emb(self, input_ids):
    out = nn.functional.embedding(input_ids, self.weight).float() * scale
    if NOISE["on"]:
        g = torch.Generator().manual_seed(0)
        out = out * (1 + EPS * torch.randn(out.shape, generator=g))
    return out


emb.forward = _emb.__get__(emb)
routers = [x for x in m.modules() if isinstance(x, Router)]
for i, x in enumerate(routers):
    x._layer = i
layers = m.model.language_model.layers if kind == "gemma4" else m.model.layers

# 1-2. full forwards capturing each layer's input and call arguments
CAP = {"x": {}, "kw": {}}
def cap(mod, args, kw, i):
    CAP["x"][i] = (args[0] if args else kw["hidden_states"]).detach().clone()
    CAP["kw"][i] = (args[1:], {k: v for k, v in kw.items() if k != "hidden_states"})
hooks = [l.register_forward_pre_hook(lambda mod, a, k, i=i: cap(mod, a, k, i), with_kwargs=True) for i, l in enumerate(layers)]
t = time.time()
with torch.no_grad():
    m(ids, use_cache=False)
clean_x, call = dict(CAP["x"]), dict(CAP["kw"])
clean_idx = {k: v.clone() for k, v in PICKED.items()}
FREEZE.update(clean_idx); NOISE["on"] = True; CAP["x"] = {}
with torch.no_grad():
    m(ids, use_cache=False)
noisy_x = dict(CAP["x"])
FREEZE.clear(); NOISE["on"] = False
for h in hooks:
    h.remove()
print(f"full forwards done in {time.time()-t:.0f}s", flush=True)


def rel(a, b):  # per position |a-b| / |b|, positions T0..
    a = a.reshape(-1, a.shape[-1])[T0:].double(); b = b.reshape(-1, b.shape[-1])[T0:].double()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30))


def run_layer(i, x, freeze):
    rec, hs = {}, []
    for n, sub in layers[i].named_modules():
        if not n:
            continue
        def hk(mod, a, o, n=n):
            rec.setdefault(n, []).append(o)
        hs.append(sub.register_forward_hook(hk))
    FREEZE.clear()
    if freeze:
        FREEZE[i] = clean_idx[i]
    a, k = call[i]
    with torch.no_grad():
        out = layers[i](x, *a, **k)
    for h in hs:
        h.remove()
    FREEZE.clear()
    pts = {"layer_out": out[0] if isinstance(out, tuple) else out}
    for n, outs in rec.items():
        if len(outs) != 1:
            continue  # called more than once (e.g. activation inside the expert loop)
        o = outs[0]
        if isinstance(o, tuple):
            if n == "self_attn" and len(o) > 1 and torch.is_tensor(o[1]):
                pts["self_attn.weights"] = o[1]
            o = o[0]
        if torch.is_tensor(o) and o.is_floating_point() and o.dim() >= 2:
            pts[n] = o
    upd = "post_attention_layernorm" if kind == "gemma4" else "self_attn"
    pts["resid_after_attn"] = x.float() + pts[upd].float()
    return pts, PICKED[i].clone()


results = []
gen = torch.Generator().manual_seed(1)
for i in range(len(layers)):
    t = time.time()
    x = clean_x[i].float()
    C, cidx = run_layer(i, x, freeze=True)
    row = {"layer": i,
           "update_share_attn": round((C["resid_after_attn"] - x).reshape(-1, x.shape[-1])[T0:].norm(dim=-1).div(x.reshape(-1, x.shape[-1])[T0:].norm(dim=-1)).mean().item(), 4),
           "out_over_in_norm": round(C["layer_out"].reshape(-1, x.shape[-1])[T0:].norm(dim=-1).div(x.reshape(-1, x.shape[-1])[T0:].norm(dim=-1)).mean().item(), 4)}
    for dname in ("random", "drift"):
        if dname == "random":
            d = torch.randn(x.shape, generator=gen) * x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5
        else:
            d = noisy_x[i].float() - x
        d = d * (EPS / rel(x + d, x).mean().item())  # mean relative size EPS
        e_in = rel(x + d, x).mean().item()
        for frz in (True, False):
            Pp, pidx = run_layer(i, x + d, freeze=frz)
            tag = f"{dname}_{'frozen' if frz else 'free'}"
            if frz:
                row[tag] = {n: round(rel(Pp[n].float(), C[n].float()).mean().item() / e_in, 3) for n in C if n in Pp and Pp[n].shape == C[n].shape}
            else:
                flips = (torch.sort(pidx, -1).values != torch.sort(cidx, -1).values).any(-1).float()
                row[tag] = {"layer_out": round(rel(Pp["layer_out"].float(), C["layer_out"].float()).mean().item() / e_in, 3),
                            "flip_pct": round(flips.mean().item() * 100, 2)}
    results.append(row)
    print("GAIN " + json.dumps(row), flush=True)
    print(f"layer {i} in {time.time()-t:.0f}s", flush=True)
json.dump(results, open(f"{DATA}/probe_layer_gain_{name}.json", "w"), indent=1)
print("DONE", name, flush=True)
