# Hugging Face Gemma-4-26B-A4B reference, fp32 arithmetic (bf16-stored weights upcast at use), on the first
# NTOK tokens of the book: for every layer save its input, attention output (o_proj), shared-MLP output,
# expert-block output, the router's chosen experts, and the layer output. Plus final logits.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import bz2, sys, torch, torch.nn as nn
torch.set_num_threads(8)
import transformers.models.gemma4.modeling_gemma4 as G
from transformers import AutoModelForCausalLM

NTOK = 512
P = f"{MODELS}/gemma-4-26B-A4B-it"
D = f"{DATA}"
ids = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][:, :NTOK]

rec = {"in": {}, "attn": {}, "mlp": {}, "experts": {}, "out": {}, "idx": {}}


def router_fwd(self, hidden_states):  # = HF Gemma4TextRouter.forward, fp32
    h = self.norm(hidden_states.float())
    h = h * self.scale.float() * self.scalar_root_size
    probs = torch.softmax(nn.functional.linear(h, self.proj.weight.float()), dim=-1)
    w, idx = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    w = w * self.per_expert_scale.float()[idx]
    rec["idx"][self._layer] = idx.detach().clone()
    return probs, w, idx


def experts_fwd(self, hidden_states, top_k_index, top_k_weights):  # = HF loop, fp32
    hidden_states = hidden_states.float()
    final = torch.zeros_like(hidden_states)
    mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e[0]
        pos, tok = torch.where(mask[e])
        gate, up = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e].float()).chunk(2, dim=-1)
        h = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e].float())
        final.index_add_(0, tok, h * top_k_weights[tok, pos, None].float())
    rec["experts"][self._layer] = final.detach().clone()
    return final


G.Gemma4TextRouter.forward = router_fwd
G.Gemma4TextExperts.forward = experts_fwd
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa").eval()


def _lin(self, x):
    return nn.functional.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


for mod in m.modules():
    if isinstance(mod, nn.Linear):
        mod.forward = _lin.__get__(mod)
emb = m.model.language_model.embed_tokens
emb.forward = (lambda self, x: nn.functional.embedding(x, self.weight).float() * float(self.scalar_embed_scale)).__get__(emb)
for i, layer in enumerate(m.model.language_model.layers):
    layer.router._layer = i
    layer.experts._layer = i
    layer.register_forward_pre_hook(lambda mod, args, kw, i=i: rec["in"].__setitem__(i, args[0].detach().clone() if args else kw["hidden_states"].detach().clone()), with_kwargs=True)
    layer.register_forward_hook(lambda mod, args, out, i=i: rec["out"].__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach().clone()))
    layer.self_attn.register_forward_hook(lambda mod, args, out, i=i: rec["attn"].__setitem__(i, out[0].detach().clone()))
    layer.mlp.register_forward_hook(lambda mod, args, out, i=i: rec["mlp"].__setitem__(i, out.detach().clone()))
with torch.no_grad():
    logits = m(ids).logits[0].float()
save = {k: torch.stack([v[i].reshape(-1, v[i].shape[-1]).float() for i in sorted(v)]) if k != "idx" else torch.stack([v[i] for i in sorted(v)]) for k, v in rec.items()}
save["logits"] = logits
save["ids"] = ids[0]
torch.save(save, f"{D}/hf_per_layer_ref_{NTOK}.pt")
print("SAVED", {k: tuple(v.shape) for k, v in save.items()}, flush=True)
