"""Gate (a): (1) single-layer teacher-forced: HF fp32 layer vs oracle fp32 / oracle bf16 on the REAL bf16-chain hidden state h_in of L0,L1,L5 (and L2-4),
PCC and router sets; (2) short chain L0..L5 HF fp32 (HF layers chained) vs oracle fp32 chain dump: per-layer h_out PCC, then final norm+lm_head logits PCC and
greedy tokens (all 2048 positions) HF vs oracle. Needs the dumps from ref_prefill_dump (bf16 dir, fp32 dir)."""
import os
import sys

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

from models.demos.blackhole.mimo_v26.reference import hf_layer as H
from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R
from models.demos.blackhole.mimo_v26.reference.ref_prefill_dump import pcc

torch.set_num_threads(32)
D16, D32 = sys.argv[1], sys.argv[2]
ck = Ld.Ckpt()
cfg = H.hf_config()
toks = torch.load(f"{D16}/tokens.pt")["prefill_tokens"]
E = ck.get("model.embed_tokens.weight")
h_hf = E[toks].float()
tm = H.masks_and_rope(cfg, h_hf)
rows = []
for L in range(6):
    sd = Ld.load_layer(L, torch.float32, ck)
    hl = H.build_layer(L, torch.float32, sd=sd, cfg=cfg)
    d16 = torch.load(f"{D16}/layer_{L}.pt")["prefill"]
    d32 = torch.load(f"{D32}/layer_{L}.pt")["prefill"]
    x = d16["h_in"].float()
    ho = H.run_layer(hl, x, cfg, tm)
    o32 = R.layer_forward(x, sd, L)
    r = {"L": L, "tf_oracle32_vs_hf_pcc": pcc(o32["h_out"], ho)}
    if L > 0:
        with torch.no_grad():
            ti, _ = hl.mlp.gate(o32["ffn_in"])
        r["router_sets_equal_frac"] = (
            (ti.sort(-1).values == o32["routing_idx"].sort(-1).values).all(-1).float().mean().item()
        )
    r["tf_bf16dump_vs_hf_pcc"] = pcc(d16["h_out"], ho)  # bf16 oracle layer output (dumped) vs HF fp32 on the same input
    # chain: HF layers chained on their own outputs vs fp32 oracle chain dump
    h_hf = H.run_layer(hl, h_hf, cfg, tm)
    r["chain_hf_vs_oracle32_pcc"] = pcc(h_hf, d32["h_out"])
    r["chain_hf_vs_bf16dump_pcc"] = pcc(h_hf, d16["h_out"])
    rows.append(r)
    print(r, flush=True)
    del sd, hl
W = ck.get("model.norm.weight")
head = ck.get("lm_head.weight").float()
lg_hf = R.rms_norm(h_hf, W).reshape(-1, 4096) @ head.T
h32 = torch.load(f"{D32}/layer_5.pt")["prefill"]["h_out"]
lg_or = R.rms_norm(h32, W).reshape(-1, 4096) @ head.T
h16 = torch.load(f"{D16}/layer_5.pt")["prefill"]["h_out"].float()
lg_16 = R.rms_norm(h16, W).reshape(-1, 4096) @ head.T
print(
    "logits after L0..L5 (prefix logits, not language): PCC hf vs oracle32 %.6f ; greedy equal %.4f ; PCC hf vs bf16 dump %.6f greedy equal %.4f"
    % (
        pcc(lg_hf, lg_or),
        (lg_hf.argmax(-1) == lg_or.argmax(-1)).float().mean(),
        pcc(lg_hf, lg_16),
        (lg_hf.argmax(-1) == lg_16.argmax(-1)).float().mean(),
    )
)
