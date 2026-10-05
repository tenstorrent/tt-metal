# Baseline for tt_per_layer_pcc.py: the same per-layer measurement for Hugging Face's own plain bf16 run
# (weights bf16, every operation in bf16, HF default), against the fp32 reference in hf_per_layer_ref_512.pt.
#   accumulated: normal bf16 forward, each layer gets bf16 HF's own previous output
#   isolated:    each layer's input is replaced by the fp32 reference input for that layer (rounded to bf16)
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import json, sys, torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM

P = f"{MODELS}/gemma-4-26B-A4B-it"
D = f"{DATA}"
ref = torch.load(f"{D}/hf_per_layer_ref_512.pt")
ids = ref["ids"].reshape(1, -1)
MODE = {"isolated": False}
rec = {}


def pre(mod, args, kw, i):
    if MODE["isolated"]:
        x = ref["in"][i].reshape(1, -1, ref["in"].shape[-1]).to(torch.bfloat16)
        if args:
            args = (x,) + tuple(args[1:])
        else:
            kw["hidden_states"] = x
    rec.setdefault("in", {})[i] = (args[0] if args else kw["hidden_states"]).detach().float().reshape(-1, ref["in"].shape[-1])
    return args, kw


def put(kind, i, t):
    rec.setdefault(kind, {})[i] = t.detach().float().reshape(-1, t.shape[-1])


m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
for i, layer in enumerate(m.model.language_model.layers):
    layer.register_forward_pre_hook(lambda mod, a, k, i=i: pre(mod, a, k, i), with_kwargs=True)
    layer.register_forward_hook(lambda mod, a, o, i=i: put("out", i, o[0] if isinstance(o, tuple) else o))
    layer.self_attn.register_forward_hook(lambda mod, a, o, i=i: put("attn", i, o[0]))
    layer.mlp.register_forward_hook(lambda mod, a, o, i=i: put("mlp", i, o))
    layer.experts.register_forward_hook(lambda mod, a, o, i=i: put("experts", i, o))
    layer.router.register_forward_hook(lambda mod, a, o, i=i: rec.setdefault("idx", {}).__setitem__(i, o[2].detach().clone()))


def pcc_rows(a, b):
    a, b = a.double(), b.double()
    a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


results = {}
for mode, iso in (("hf_bf16_accumulated", False), ("hf_bf16_isolated", True)):
    rec.clear(); MODE["isolated"] = iso
    with torch.no_grad():
        m(ids)
    out = []
    for i in range(len(rec["out"])):
        r = {"layer": i}
        for kind in ("in", "attn", "mlp", "experts", "out"):
            p = pcc_rows(rec[kind][i], ref[kind][i])
            r[kind] = round(p.mean().item(), 5)
            r[kind + "_min"] = round(p.min().item(), 4)
        a = torch.zeros(ids.shape[1], 128, dtype=torch.bool); a.scatter_(1, rec["idx"][i], True)
        b = torch.zeros_like(a); b.scatter_(1, ref["idx"][i], True)
        overlap = (a & b).sum(-1).float()
        r["experts_same_8"] = round((overlap == 8).float().mean().item() * 100, 1)
        r["experts_overlap_of_8"] = round(overlap.mean().item(), 3)
        out.append(r)
        print(f"{mode.upper()} " + json.dumps(r), flush=True)
    results[mode] = out
json.dump(results, open(f"{D}/hf_bf16_per_layer_pcc.json", "w"), indent=1)
print("DONE", flush=True)
