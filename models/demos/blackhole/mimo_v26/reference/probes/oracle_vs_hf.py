"""Mask convention (HF create_sliding_window_causal_mask vs oracle) and fp32 oracle (ref_layer.py) vs the HF decoder layer for L0/L1/L5, router sets."""
import os

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

torch.set_num_threads(32)
from models.demos.blackhole.mimo_v26.reference import hf_layer as H
from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R

ck = Ld.Ckpt()
cfg = H.hf_config()
mod, _ = H.import_hf()
pcc = lambda a, b: torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
from models.demos.blackhole.mimo_v26.reference import prompts as P

t, _ = P.gsm8k_tokens(2, 128)
x = ck.get("model.embed_tokens.weight")[t].float()
tm = H.masks_and_rope(cfg, x)
# mask convention
S = 300
m = H.masks_and_rope(cfg, torch.zeros(1, S, 4096))[0]
for name, swa in (("full_attention", False), ("sliding_window_attention", True)):
    hm = m[name][0, 0]
    vis = (hm == 0) if hm.dtype != torch.bool else hm
    mine = R.attn_mask(S, S, swa)
    print(
        name,
        "HF mask == oracle mask:",
        torch.equal(vis, mine),
        "| row 200 visible span",
        vis[200].nonzero().min().item(),
        vis[200].nonzero().max().item(),
        "count",
        vis[200].sum().item(),
    )
for L in (0, 1, 5):
    sd = Ld.load_layer(L, torch.float32, ck)
    hl = H.build_layer(L, torch.float32, sd=sd, cfg=cfg)
    xin = x
    with torch.no_grad():
        ho = H.run_layer(hl, xin, cfg, tm)
    r = R.layer_forward(xin, sd, L)
    print(
        "L",
        L,
        "fp32 oracle vs HF h_out PCC %.6f maxabs %.3e rel %.2e"
        % (pcc(r["h_out"], ho), (r["h_out"] - ho).abs().max(), ((r["h_out"] - ho).norm() / ho.norm())),
        flush=True,
    )
    if L > 0:
        with torch.no_grad():
            ti, tw = hl.mlp.gate(r["ffn_in"])
        a = ti.sort(-1).values
        b = r["routing_idx"].sort(-1).values
        print(
            "  router set equal on all tokens:",
            torch.equal(a, b.view_as(a)),
            " weights maxabs diff",
            (tw.gather(1, ti.argsort(-1)) - r["routing_w"].gather(1, r["routing_idx"].argsort(-1))).abs().max().item()
            if False
            else "",
        )
