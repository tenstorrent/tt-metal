"""Probe: how much of the MoE error comes from bfp4_b re-quantisation of (a) routed experts, (b) the shared expert?
Run directly (python), single device."""
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

L = 2
dev = ttnn.open_device(device_id=0)
rt = (
    lambda w: ttnn.to_torch(
        ttnn.from_torch(
            w.float().reshape(1, 1, *w.shape[-2:]), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev
        )
    )
    .reshape(w.shape)
    .float()
)
rt8 = (
    lambda w: ttnn.to_torch(
        ttnn.from_torch(
            w.float().reshape(1, 1, *w.shape[-2:]), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev
        )
    )
    .reshape(w.shape)
    .float()
)

blk = R.build_layer(L, max_batch_size=16, max_seq_len=64)
torch.manual_seed(0)
h, pm = R.embed_tokens(torch.randint(1000, 100000, (16, 1)))
cap = {}
blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
blk(h, 0, pm, None)
x = cap["x"].reshape(16, 5120).float()
w = load_moe_layer(L)
s = torch.nn.functional.softplus(x @ w["gate_weight"]).sqrt()
idx = (s + w["gate_bias"]).topk(6, -1).indices
wt = s.gather(1, idx)
wt = wt / wt.sum(-1, keepdim=True) * 1.5


def expert(x1, a, b, c, rtf):
    a, b, c = (rtf(m) for m in (a, b, c))
    return (torch.nn.functional.silu(x1 @ a) * (x1 @ b)) @ c


def moe(rt_routed, rt_shared):
    out = torch.zeros(16, 5120)
    for t in range(16):
        xt = x[t : t + 1]
        for j in range(6):
            e = int(idx[t, j])
            out[t] += (
                wt[t, j] * expert(xt, w["w0"][0, e].float(), w["w1"][0, e].float(), w["w2"][0, e].float(), rt_routed)[0]
            )
        out[t] += expert(
            xt,
            w["shared_w0"][384][0, 0].float(),
            w["shared_w1"][384][0, 0].float(),
            w["shared_w2"][384][0, 0].float(),
            rt_shared,
        )[0]
    return out


ident = lambda m: m


def hilo(m):  # two bfp4 terms: Q(W) + Q(W - Q(W))
    hi = rt(m)
    return hi + rt(m.float() - hi)


exact = moe(ident, ident)
for name, a, b in (("routed bfp4 only", rt, ident), ("routed hi+lo bfp4", hilo, ident), ("routed bfp8", rt8, ident)):
    y = moe(a, b)
    print(
        f"RESULT {name:28s} MoE PCC vs exact {R.pcc(y, exact):.5f}  rel err {((y - exact).norm() / exact.norm()):.4f}"
    )
ttnn.close_device(dev)
