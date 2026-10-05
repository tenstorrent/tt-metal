# Which part of the model turns a tiny difference into a large one? Same 2^-9 embedding nudge as cross_model_perturb6.py
# (any HF MoE: Gemma 4, Qwen3-MoE, OLMoE; weights stored bf16, all arithmetic fp32, eager attention, 1023 book tokens),
# run with parts of the model frozen to the clean run's values:
#   noise                 - nudge only
#   noise_frozen_routing  - every router's expert choice forced to the clean run's
#   noise_frozen_attn     - every layer's attention probabilities (after softmax) forced to the clean run's
#   noise_frozen_both     - both
# Reports PCC of the final logits vs the clean run over positions 511-1010, the relative difference of the residual
# stream and of the first norm's output ("what the layer reads") at every layer, and the expert-flip rate by layer.
# FREEZE_SUBSETS=1 instead freezes attention in only some layers (global / sliding / each third of the depth).
# Usage: freeze_components.py <model_dir> <gemma4|qwen3_moe|olmoe>
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, gc, json, sys, time
import torch
import torch.nn as nn
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

P, kind = sys.argv[1], sys.argv[2]
name = P.rstrip("/").split("/")[-1]
NTOK, EPS = 1023, 2.0 ** -9
if kind == "gemma4":
    import transformers.models.gemma4.modeling_gemma4 as M
    Router, Experts = M.Gemma4TextRouter, M.Gemma4TextExperts
elif kind == "qwen3_moe":
    import transformers.models.qwen3_moe.modeling_qwen3_moe as M
    Router, Experts = M.Qwen3MoeTopKRouter, M.Qwen3MoeExperts
else:
    import transformers.models.olmoe.modeling_olmoe as M
    Router, Experts = M.OlmoeTopKRouter, M.OlmoeExperts

F = {}
REF_IDX, PICKED, REF_ATTN = {}, {}, {}


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
    if F.get("freeze_routing"):
        idx = REF_IDX[self._layer]
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


_eager = M.eager_attention_forward
def eager_fwd(module, query, key, value, attention_mask, *a, **k):
    out, w = _eager(module, query, key, value, attention_mask, *a, **k)
    i = module.layer_idx
    if F.get("record"):
        REF_ATTN[i] = w.detach().clone()
    sub = F.get("attn_layers")  # None = every layer
    if F.get("freeze_attn") and (sub is None or i in sub):  # the eager function's last steps, with the clean run's probabilities
        w = REF_ATTN[i]
        out = torch.matmul(w, M.repeat_kv(value, module.num_key_value_groups)).transpose(1, 2).contiguous()
    return out, w


Router.forward = router_fwd
Experts.forward = experts_fwd
M.eager_attention_forward = eager_fwd

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
    if F.get("noise"):
        g = torch.Generator().manual_seed(0)
        out = out * (1 + EPS * torch.randn(out.shape, generator=g))
    return out


emb.forward = _emb.__get__(emb)
routers = [x for x in m.modules() if isinstance(x, Router)]
for i, x in enumerate(routers):
    x._layer = i
layers = m.model.language_model.layers if kind == "gemma4" else m.model.layers
NORMED = {}
for i, l in enumerate(layers):
    l.input_layernorm.register_forward_hook(lambda mod, a, o, i=i: NORMED.__setitem__(i, o[0, 511:1011].float().clone()))

CONFIGS = [("clean", {"record": True}), ("noise", {"noise": True}), ("noise_frozen_routing", {"noise": True, "freeze_routing": True}),
           ("noise_frozen_attn", {"noise": True, "freeze_attn": True}), ("noise_frozen_both", {"noise": True, "freeze_routing": True, "freeze_attn": True})]
if _os.environ.get("FREEZE_SUBSETS") == "1":  # attention frozen in only some layers
    n = len(layers)
    glob = [i for i, l in enumerate(layers) if not getattr(l.self_attn, "is_sliding", True)]
    CONFIGS = CONFIGS[:2] + [
        ("noise_frozen_attn_global_layers", {"noise": True, "freeze_attn": True, "attn_layers": set(glob)}),
        ("noise_frozen_attn_sliding_layers", {"noise": True, "freeze_attn": True, "attn_layers": set(range(n)) - set(glob)}),
        ("noise_frozen_attn_first_third", {"noise": True, "freeze_attn": True, "attn_layers": set(range(0, n // 3))}),
        ("noise_frozen_attn_middle_third", {"noise": True, "freeze_attn": True, "attn_layers": set(range(n // 3, 2 * n // 3))}),
        ("noise_frozen_attn_last_third", {"noise": True, "freeze_attn": True, "attn_layers": set(range(2 * n // 3, n))})]
s = slice(511, 1011)
rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()
results = {}
for cname, flags in CONFIGS:
    F.clear(); F.update(flags); PICKED.clear(); NORMED.clear()
    t0 = time.time()
    with torch.no_grad():
        out = m(ids, output_hidden_states=True, use_cache=False)
    lg = out.logits[0].float()
    hs = [h[0, s].float() for h in out.hidden_states]
    if cname == "clean":
        ref, ref_hs, ref_n = lg, hs, dict(NORMED)
        REF_IDX.update({k: v.clone() for k, v in PICKED.items()})
        print(f"clean run in {time.time()-t0:.0f}s", flush=True)
        continue
    pc = torch.stack([torch.corrcoef(torch.stack((lg[i], ref[i])))[0, 1] for i in range(s.start, s.stop)])
    fl = [(torch.sort(PICKED[k], -1).values != torch.sort(REF_IDX[k], -1).values).any(-1).float().mean().item() * 100 for k in sorted(PICKED)]
    r = dict(model=name, config=cname, mean_pcc_500=round(pc.mean().item(), 5), below_099=int((pc < 0.99).sum()),
             top1_agree=round((lg[s].argmax(-1) == ref[s].argmax(-1)).float().mean().item() * 100, 1),
             flip_mean=round(sum(fl) / len(fl), 2),
             residual_drift_by_layer=[round(rel(hs[i], ref_hs[i]), 5) for i in range(len(hs))],
             normed_drift_by_layer=[round(rel(NORMED[i], ref_n[i]), 5) for i in sorted(NORMED)],
             flip_pct_by_layer=[round(x, 2) for x in fl], seconds=round(time.time() - t0))
    results[cname] = r
    print("FREEZE " + json.dumps(r), flush=True)
    gc.collect()
json.dump(results, open(f"{DATA}/freeze_components_{name}{'_subsets' if _os.environ.get('FREEZE_SUBSETS') == '1' else ''}.json", "w"), indent=1)
print("DONE", name, flush=True)
