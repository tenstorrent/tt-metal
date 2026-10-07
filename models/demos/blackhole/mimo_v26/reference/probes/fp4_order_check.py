"""fp4 expert check on real layer-1 experts: nibble order by routing selectivity (experts must respond more to tokens the router sends them), e8m0 bias plausibility."""
import os

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

torch.set_num_threads(32)
import torch.nn.functional as F

from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import prompts as P
from models.demos.blackhole.mimo_v26.reference import ref_layer as R

ck = Ld.Ckpt()
t, _ = P.gsm8k_tokens(16, 128)
E = ck.get("model.embed_tokens.weight")
h = E[t].float()
sd0 = Ld.load_layer(0, torch.float32, ck)
r0 = R.layer_forward(h, sd0, 0)
sd1 = Ld.load_layer(1, torch.float32, ck, experts=False)
r1 = R.layer_forward(r0["h_out"], {**sd1}, 1) if False else None
# attention + router of L1 only
x = r0["h_out"]
a = R.attention(R.rms_norm(x, sd1["input_layernorm.weight"]), sd1, 1)[0]
h1 = x + a
fi = R.rms_norm(h1, sd1["post_attention_layernorm.weight"]).reshape(-1, 4096)
idx, w, gap = R.router(fi, sd1)
cnt = torch.bincount(idx.flatten(), minlength=256)
print("tokens", fi.shape[0], "expert load min/med/max", cnt.min().item(), cnt.median().item(), cnt.max().item())


def deq(raw, sc, order, bias=127):
    b = raw.view(torch.uint8)
    lo, hi = Ld.FP4_TABLE[(b & 15).long()], Ld.FP4_TABLE[(b >> 4).long()]
    pair = torch.stack([lo, hi] if order == "lo_first" else [hi, lo], -1)
    n = b.size(0)
    s = torch.pow(2.0, sc.view(torch.uint8).float() - bias)
    return (pair.reshape(n, -1).view(n, -1, 32) * s.unsqueeze(-1)).reshape(n, -1)


res = {}
for e in [3, 17, 40, 100, 200]:
    sel = (idx == e).any(1)
    if sel.sum() < 8:
        continue
    row = {}
    for order in ("lo_first", "hi_first"):
        g = (
            fi
            @ deq(
                ck.get(f"model.layers.1.mlp.experts.{e}.gate_proj.weight"),
                ck.get(f"model.layers.1.mlp.experts.{e}.gate_proj.weight_scale"),
                order,
            ).T
        )
        u = (
            fi
            @ deq(
                ck.get(f"model.layers.1.mlp.experts.{e}.up_proj.weight"),
                ck.get(f"model.layers.1.mlp.experts.{e}.up_proj.weight_scale"),
                order,
            ).T
        )
        act = (F.silu(g) * u).norm(dim=1)
        row[order] = (act[sel].mean() / act[~sel].mean()).item()
    print(
        "expert",
        e,
        "routed tokens",
        sel.sum().item(),
        "act-norm ratio routed/unrouted:",
        {k: round(v, 3) for k, v in row.items()},
    )
# scale/bias plausibility
raw = ck.get("model.layers.1.mlp.experts.3.gate_proj.weight")
sc = ck.get("model.layers.1.mlp.experts.3.gate_proj.weight_scale")
print(
    "e8m0 byte min/max", sc.min().item(), sc.max().item(), "-> exponents", sc.min().item() - 127, sc.max().item() - 127
)
for bias in (126, 127, 128):
    print("bias", bias, "expert3 gate rms %.5f" % deq(raw, sc, "lo_first", bias).pow(2).mean().sqrt().item())
print(
    "dense L0 gate rms %.5f" % sd0["mlp.gate_proj.weight"].pow(2).mean().sqrt().item(),
    "L1 qkv rms %.5f" % sd1["self_attn.qkv_proj.weight"].pow(2).mean().sqrt().item(),
)
# top block code check: per block max |code value| distribution
b = raw.view(torch.uint8)
lo, hi = Ld.FP4_TABLE[(b & 15).long()], Ld.FP4_TABLE[(b >> 4).long()]
mx = torch.stack([lo, hi], -1).reshape(2048, -1).view(2048, -1, 32).abs().amax(-1)
print("per-block max |code|: frac==6 %.3f frac>=4 %.3f" % ((mx == 6).float().mean(), (mx >= 4).float().mean()))
# pure e8m0 view vs 2**(b-127)
print("e8m0 view == 2**(b-127):", torch.equal(sc.view(torch.float8_e8m0fnu).float(), torch.pow(2.0, sc.float() - 127)))
