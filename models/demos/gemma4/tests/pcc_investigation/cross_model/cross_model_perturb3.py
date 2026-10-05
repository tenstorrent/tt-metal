# Same perturbation experiment on any HF MoE (Gemma 4, Qwen3-MoE, OLMoE): weights stored bf16,
# all arithmetic fp32 (upcast at use), 1023 book tokens. Configs:
#   none           - reference, nothing rounded
#   emb            - embedding output rounded to bf16 once (a tiny nudge at the input)
#   router_logits  - router input + scores rounded to bf16, softmax/top-k in fp32 (what HF does for Qwen3/OLMoE)
#   router_probs   - router_logits + probabilities rounded to bf16 before top-k (what HF does for Gemma 4)
# Reports PCC vs reference over positions 511-1010, expert-flip rates by depth, and router near-tie stats.
# Usage: cross_model_perturb.py <model_dir> <gemma4|qwen3_moe|olmoe>
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
if kind == "gemma4":
    import transformers.models.gemma4.modeling_gemma4 as M
    Router, Experts = M.Gemma4TextRouter, M.Gemma4TextExperts
elif kind == "qwen3_moe":
    import transformers.models.qwen3_moe.modeling_qwen3_moe as M
    Router, Experts = M.Qwen3MoeTopKRouter, M.Qwen3MoeExperts
else:
    import transformers.models.olmoe.modeling_olmoe as M
    Router, Experts = M.OlmoeTopKRouter, M.OlmoeExperts

F, rec, gaps, sstats, calls = {}, {}, {}, {}, {"router": 0, "experts": 0}
r = lambda x: x.to(torch.bfloat16).to(x.dtype)


def router_fwd(self, hidden_states):
    calls["router"] += 1
    rl = r if F.get("router_logits") or F.get("router_probs") else (lambda x: x)
    rp = r if F.get("router_probs") else (lambda x: x)
    if kind == "gemma4":
        h = rl(self.norm(hidden_states.float()))
        h = rl(rl(h * rl(self.scale.float())) * self.scalar_root_size)
        k = self.config.top_k_experts
    else:
        h = rl(hidden_states.reshape(-1, self.hidden_dim).float())
        k = self.top_k
    logits = rl(nn.functional.linear(h, self.weight.float() if kind != "gemma4" else self.proj.weight.float()))
    probs = rp(torch.softmax(logits, dim=-1, dtype=torch.float32))
    w, idx = torch.topk(probs, k, dim=-1)
    lt = logits.float().topk(k + 1, -1).values
    sstats[self._layer] = (logits.float().abs().mean(-1), lt[:, k - 1], lt[:, k - 1] - lt[:, k])
    if F.get("swap_layer") == self._layer:  # force: replace the 8th-best expert by the 9th-best
        top = torch.topk(probs, k + 1, dim=-1)
        idx = torch.cat([top.indices[:, : k - 1], top.indices[:, k:k + 1]], -1)
        w = probs.gather(-1, idx)
    t9 = probs.topk(k + 1, -1).values
    gaps[self._layer] = (t9[:, k - 1] - t9[:, k]).detach(), t9[:, k - 1].detach()
    if kind == "gemma4":
        w = w / w.sum(-1, keepdim=True)
        w = w * self.per_expert_scale.float()[idx]
        rec[self._layer] = idx.detach()
        return probs, w, idx
    if self.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    rec[self._layer] = idx.detach()
    return logits, w, idx


def experts_fwd(self, hidden_states, top_k_index, top_k_weights):
    calls["experts"] += 1
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
ids = torch.tensor([ids[:1023]])

m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa").eval()


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
        out = out * (1 + 2.0 ** -9 * torch.randn(out.shape, generator=g))
    return r(out) if F.get("emb") else out


emb.forward = _emb.__get__(emb)
routers = [x for x in m.modules() if isinstance(x, Router)]
for i, x in enumerate(routers):
    x._layer = i

CONFIGS = [("none", {}), ("noise", {"noise": True}), ("swap_first_layer", {"swap_layer": 0})]
s = slice(511, 1011)
ref = None
for cname, flags in CONFIGS:
    F.clear(); F.update(flags); rec.clear(); gaps.clear(); sstats.clear()
    t0 = time.time()
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
        lg = out.logits[0].float()
        hs = [h[0, 511:1011].float() for h in out.hidden_states]
    if ref is None:
        ref, ref_rec, ref_hs = lg, {k: torch.sort(v, -1).values for k, v in rec.items()}, hs
        g = torch.cat([gaps[k][0] for k in sorted(gaps)]); p8 = torch.cat([gaps[k][1] for k in sorted(gaps)])
        # bf16 spacing at the 8th probability: 2^(floor(log2 p8) - 7)
        step = torch.pow(2.0, torch.floor(torch.log2(p8)) - 7)
        print("TIES " + json.dumps(dict(model=name, layers=len(routers), experts=routers[0].num_experts if hasattr(routers[0], "num_experts") else None,
              median_p8=round(p8.median().item(), 5), median_gap_8_9=round(g.median().item(), 6),
              share_gap_below_1_bf16_step=round((g < step).float().mean().item() * 100, 2),
              share_gap_below_4_bf16_steps=round((g < 4 * step).float().mean().item() * 100, 2),
              router_calls=calls["router"], expert_calls=calls["experts"], seconds=round(time.time() - t0))), flush=True)
        absmean = torch.cat([sstats[k][0] for k in sorted(sstats)]); s8 = torch.cat([sstats[k][1] for k in sorted(sstats)])
        lgap = torch.cat([sstats[k][2] for k in sorted(sstats)])
        lstep = torch.pow(2.0, torch.floor(torch.log2(s8.abs().clamp_min(1e-6))) - 7)  # bf16 spacing at the 8th score
        per_layer = [round(sstats[k][0].mean().item(), 2) for k in sorted(sstats)]
        print("SCORES " + json.dumps(dict(model=name, mean_abs_score=round(absmean.mean().item(), 3), median_abs_8th_score=round(s8.abs().median().item(), 3),
              median_score_gap_8_9=round(lgap.median().item(), 4), median_bf16_step_at_8th=round(lstep.median().item(), 4),
              share_score_gap_below_1_step=round((lgap < lstep).float().mean().item() * 100, 2),
              share_score_gap_below_4_steps=round((lgap < 4 * lstep).float().mean().item() * 100, 2),
              mean_abs_score_by_layer=per_layer)), flush=True)
        continue
    pc = torch.stack([torch.corrcoef(torch.stack((lg[i], ref[i])))[0, 1] for i in range(lg.shape[0])])
    fl = [(torch.sort(rec[k], -1).values != ref_rec[k]).any(-1).float().mean().item() * 100 for k in sorted(rec)]
    n = len(fl)
    drift = [round(((hs[i] - ref_hs[i]).norm(dim=-1) / ref_hs[i].norm(dim=-1)).mean().item(), 5) for i in range(len(hs))]
    print("DRIFT " + json.dumps(dict(model=name, config=cname, relative_residual_drift_by_layer=drift)), flush=True)
    print("PERTURB2 " + json.dumps(dict(model=name, config=cname, mean_pcc_500=round(pc[s].mean().item(), 5),
          below_099=int((pc[s] < 0.99).sum()), top1_agree=round((lg[s].argmax(-1) == ref[s].argmax(-1)).float().mean().item() * 100, 1),
          flip_mean=round(sum(fl) / n, 2), flip_first=round(fl[0], 2), flip_quarter=round(fl[n // 4], 2),
          flip_half=round(fl[n // 2], 2), flip_last=round(fl[-1], 2))), flush=True)
    gc.collect()
print("DONE", name, flush=True)
