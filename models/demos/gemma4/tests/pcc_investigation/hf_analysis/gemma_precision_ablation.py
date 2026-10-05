# Which bf16 rounding makes Gemma-4-26B-A4B unstable?
# Load HF Gemma 4 in fp32 (CPU, eager attention), run 1023 book tokens with NO rounding (reference),
# then re-run with bf16 rounding inserted at exactly one place, and report PCC vs the reference
# plus how often the router's chosen 8 experts differ from the reference.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, gc, json, sys, time
import torch
torch.set_num_threads(8)
import torch.nn as nn
import transformers.models.gemma4.modeling_gemma4 as G
from transformers import AutoModelForCausalLM

P = f"{MODELS}/gemma-4-26B-A4B-it"
D = f"{DATA}"
ids = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][:, :1023]

F = {}  # active rounding flags


def r(x):
    """Round to bf16 and back (emulates storing x in bf16)."""
    return x.to(torch.bfloat16).to(x.dtype)


# --- attention: same math as G.eager_attention_forward, with optional rounding points ---
calls = {"attn": 0, "experts": 0, "router": 0}


def eager_patched(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
    calls["attn"] += 1
    if scaling is None:
        scaling = module.head_dim ** -0.5
    if F.get("qk"):
        query, key = r(query), r(key)
    if F.get("v"):
        value = r(value)
    k = G.repeat_kv(key, module.num_key_value_groups)
    v = G.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if F.get("logits"):
        w = r(w)
    if softcap is not None:
        w = torch.tanh(w / softcap) * softcap
    if attention_mask is not None:
        w = w + attention_mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if F.get("probs"):
        w = r(w)
    out = torch.matmul(w, v)
    if F.get("v"):
        out = r(out)
    return out.transpose(1, 2).contiguous(), w


G.eager_attention_forward = eager_patched

# --- router: same math as G.Gemma4TextRouter.forward, optional bf16 emulation; records choices ---
rec = {}


def router_patched(self, hidden_states):
    calls["router"] += 1
    q = r if F.get("router") else (lambda x: x)
    h = q(self.norm(hidden_states))
    h = q(q(h.float() * q(self.scale.float())) * self.scalar_root_size)
    scores = q(self.proj(h))
    probs = q(nn.functional.softmax(scores, dim=-1))
    w, idx = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    w = w * self.per_expert_scale.float()[idx]
    rec[self._layer] = idx.detach()
    return probs, w, idx


G.Gemma4TextRouter.forward = router_patched

# --- experts: same math as G.Gemma4TextExperts.forward, optional bf16 emulation of every step ---
orig_experts = G.Gemma4TextExperts.forward


def experts_patched(self, hidden_states, top_k_index, top_k_weights):
    calls["experts"] += 1
    rr = r if F.get("experts") else (lambda x: x)
    hidden_states = hidden_states.float()
    final = torch.zeros_like(hidden_states)
    mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e[0]
        pos, tok = torch.where(mask[e])
        x = rr(hidden_states[tok])
        gate, up = rr(nn.functional.linear(x, self.gate_up_proj[e].float())).chunk(2, dim=-1)
        h = rr(rr(self.act_fn(gate)) * up)
        h = rr(nn.functional.linear(h, self.down_proj[e].float()))
        h = rr(h * top_k_weights[tok, pos, None].float())
        final.index_add_(0, tok, h)
        final = rr(final)
    return final


G.Gemma4TextExperts.forward = experts_patched

print("loading fp32 model", flush=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").eval()
# Weights stay stored in bf16 (the checkpoint's own values, so an fp32 copy is identical); every
# operation computes in fp32 by upcasting the weight at use. Memory ~55 GB instead of ~104 GB.
def _lin_fp32(self, x):
    return nn.functional.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())
for mod in m.modules():
    if isinstance(mod, nn.Linear):
        mod.forward = _lin_fp32.__get__(mod)
def _emb_fp32(self, input_ids):
    return nn.functional.embedding(input_ids, self.weight).float() * float(self.scalar_embed_scale)
m.model.language_model.embed_tokens.forward = _emb_fp32.__get__(m.model.language_model.embed_tokens)
lm = m.model.language_model
for i, layer in enumerate(lm.layers):
    layer.router._layer = i
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

# --- hook-based rounding: linear outputs, layer norms, residual (layer outputs), embedding ---
router_linears = {id(l.router.proj) for l in lm.layers}
layer_norm_names = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                    "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                    "pre_feedforward_layernorm_2", "post_feedforward_layernorm_2")


def hook(flag):
    def h(mod, inp, out):
        if F.get(flag):
            return r(out) if isinstance(out, torch.Tensor) else (r(out[0]),) + tuple(out[1:])
        return out
    return h


for name, mod in lm.named_modules():
    if isinstance(mod, nn.Linear) and id(mod) not in router_linears:
        mod.register_forward_hook(hook("lin"))
    if isinstance(mod, G.Gemma4RMSNorm) and name.split(".")[-1] in layer_norm_names:
        mod.register_forward_hook(hook("norm"))
for layer in lm.layers:
    layer.register_forward_hook(hook("res"))
lm.embed_tokens.register_forward_hook(hook("emb"))

CONFIGS = [
    ("none", []),
    ("qk: queries+keys bf16", ["qk"]),
    ("logits: attention scores bf16", ["logits"]),
    ("probs: attention probabilities bf16", ["probs"]),
    ("v: values + attention output bf16", ["v"]),
    ("lin: all matmul outputs bf16", ["lin"]),
    ("norm: layer-norm outputs bf16", ["norm"]),
    ("res: residual stream (layer outputs) bf16", ["res"]),
    ("router: router steps bf16", ["router"]),
    ("experts: expert steps bf16", ["experts"]),
    ("emb: embedding output bf16", ["emb"]),
    ("ALL of the above", ["qk", "logits", "probs", "v", "lin", "norm", "res", "router", "experts", "emb"]),
]
s = slice(511, 1011)
ref_logits = ref_rec = None
results = []
for name, flags in CONFIGS:
    F.clear()
    F.update({f: True for f in flags})
    rec.clear()
    calls["attn"] = 0
    t0 = time.time()
    with torch.no_grad():
        lg = m(ids).logits[0].float()
    if ref_logits is None:
        ref_logits, ref_rec = lg, {k: torch.sort(v, -1).values for k, v in rec.items()}
        print(f"reference done in {time.time()-t0:.0f}s, attention calls {calls['attn']}, router calls {calls['router']}, expert calls {calls['experts']}, logits dtype {lg.dtype}", flush=True)
        true32 = torch.load(f"{D}/book_logits_hf_fp32_eager.pt").float()
        pv = torch.stack([torch.corrcoef(torch.stack((lg[i], true32[i])))[0, 1] for i in range(lg.shape[0])])
        print(f"VALIDATION no-rounding run vs true fp32 model: mean PCC {pv.mean():.6f}, min {pv.min():.6f}, same top word {(lg.argmax(-1)==true32.argmax(-1)).float().mean()*100:.2f}%", flush=True)
        continue
    p = torch.stack([torch.corrcoef(torch.stack((lg[i], ref_logits[i])))[0, 1] for i in range(lg.shape[0])])
    flips = [(torch.sort(rec[k], -1).values != ref_rec[k]).any(-1).float().mean().item() * 100 for k in sorted(rec)]
    row = dict(config=name, mean_pcc_500=round(p[s].mean().item(), 5), below_099=int((p[s] < 0.99).sum()),
               top1_agree=round((lg[s].argmax(-1) == ref_logits[s].argmax(-1)).float().mean().item() * 100, 1),
               expert_flip_pct_mean=round(sum(flips) / len(flips), 2), expert_flip_first_layer=round(flips[0], 2),
               expert_flip_last_layer=round(flips[-1], 2), seconds=round(time.time() - t0))
    results.append(row)
    print("ABLATION " + json.dumps(row), flush=True)
    gc.collect()
json.dump(results, open(f"{D}/gemma_precision_ablation.json", "w"), indent=1)
print("DONE", flush=True)
