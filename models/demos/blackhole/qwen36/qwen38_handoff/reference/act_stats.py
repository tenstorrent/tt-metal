import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

tag, path = sys.argv[1], sys.argv[2]
torch.set_num_threads(12)
d = torch.load("/home/ttuser/atupe/qwen38_work/runs/T3/kl/hf38_ref2048.refpt")
toks = d["reference_tokens"][:, :1024]
# sanity vs 3.6 refpt
d36 = torch.load("/home/ttuser/atupe/qwen38_work/runs/T3/kl/hf36_ref1024.refpt")
print("tokens equal 36/38:", torch.equal(d36["reference_tokens"][:, :1024], toks), flush=True)
cfg = AutoConfig.from_pretrained(path)
cfg.rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}  # as in ref gen
model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16)
model.eval()


def stats(x):
    x = x.float().reshape(-1, x.shape[-1])
    a = x.abs()
    cm = a.amax(0)
    top = torch.topk(cm, 5)
    flat = a.flatten()
    k = int(flat.numel() * 0.9999)
    p = flat.kthvalue(k).values.item()
    mu = x.mean()
    sd = x.std()
    kurt = (((x - mu) ** 4).mean() / sd**4).item()
    a1 = a[1:]
    cm1 = a1.amax(0)
    t1 = torch.topk(cm1, 5)
    return dict(
        max=a.max().item(),
        top5=[round(v, 3) for v in top.values.tolist()],
        top5_ch=top.indices.tolist(),
        p9999=p,
        std=sd.item(),
        kurt=kurt,
        max_ex0=a1.max().item(),
        top5_ex0=[round(v, 3) for v in t1.values.tolist()],
        top5_ch_ex0=t1.indices.tolist(),
        p9999_ex0=a1.flatten().kthvalue(int(a1.numel() * 0.9999)).values.item(),
        std_ex0=x[1:].std().item(),
    )


norms = {}


def mk(i):
    def h(m, inp, out):
        norms[i] = stats(out)

    return h


layers = model.model.layers
for i, l in enumerate(layers):
    l.input_layernorm.register_forward_hook(mk(i))
with torch.no_grad():
    out = model.model(toks, output_hidden_states=True)
hs = out.hidden_states  # [emb, l0..l(n-1)] (last one post final norm)
res = dict(
    layer_out=[stats(hs[i + 1]) for i in range(len(layers))],
    ln_in=[norms[i] for i in range(len(layers))],
    layer_types=[
        getattr(l, "layer_type", None) or type(getattr(l, "linear_attn", getattr(l, "self_attn", None))).__name__
        for l in layers
    ],
)
json.dump(res, open(f"/home/ttuser/atupe/qwen38_work/runs/T3/bisect/act_stats_{tag}.json", "w"))
print("done", tag)
