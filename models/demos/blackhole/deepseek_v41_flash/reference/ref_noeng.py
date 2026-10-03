# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference with the Engram contribution removed (all layers / only layer 1 / only layer 14), TEACHER-FORCED along the full-Engram reference
continuation of ``ref_e2e_gen`` (same GSM8K prompts), plus the gate / value statistics of the Engram at layers 1 and 14.
Saves /mnt/tt-data/ssinghal/dsv4-e2e-ref/noeng_<variant>.pt: {topk_val/idx [n,B,50], lse [n,B], full [min(n,48),B,V] bf16, gate stats}.
Run: python -m ...reference.ref_noeng --variant none|e1|e14|both --B 16 --steps 48
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels as K
from models.demos.blackhole.deepseek_v41_flash.reference import ref_spec_accept as SA

REF = "/mnt/tt-data/ssinghal/dsv4-e2e-ref"


@torch.no_grad()
def engram_stats(e, x, hash_ids):
    """gate (sigmoid of the signed-sqrt dot) and |gate * value| relative to |h| for one application (math of Engram.forward)"""
    kv = e.wkv(e.embed(hash_ids).flatten(-2))
    key, value = kv.split([e.hc_mult * e.dim, e.dim], dim=-1)
    key = key.float().unflatten(-1, (e.hc_mult, e.dim))
    weight = e.q_weight.float() * e.k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + e.eps) * torch.rsqrt(key.square().mean(-1) + e.eps)
    dot = (h * weight * key).sum(-1) * rstd * e.dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(e.clamp_value).sqrt(), dot))
    add = gate.unsqueeze(-1) * value.float().unsqueeze(-2)
    rel = add.norm(dim=-1) / (h.norm(dim=-1) + 1e-30)  # [B,L,hc]
    return gate.flatten(), rel.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="none", choices=["none", "e1", "e14", "both"])
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--steps", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    K.FAKE_QUANT = False
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    meta = torch.load(f"{REF}/meta.pt")
    prompt, S = meta["prompt"][: a.B], meta["S"]
    keep = {"none": (), "e1": (1,), "e14": (14,), "both": (1, 14)}[a.variant]
    torch.set_default_dtype(torch.bfloat16)
    from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEmbedding, HostEngram, HostHead

    emb, head = HostEmbedding(), HostHead()
    max_seq = S + a.steps + 8
    engram = HostEngram(max_batch_size=a.B, max_seq_len=max_seq)
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(SA.build_backbone_layer, L, a.B, max_seq) for L in range(40)]
        blocks = [f.result() for f in futs]
    log("built")
    stats = {1: ([], []), 14: ([], [])}

    def backbone(tokens, start_pos, record):
        h, pm = emb(tokens)
        hashes = engram.hashes(tokens, start_pos)
        for L, blk in enumerate(blocks):
            if L in engram.mods:
                hh = hashes[:, :, engram.mods[L].layer_hash_index, :]
                if record:
                    g, r = engram_stats(engram.mods[L], h, hh)
                    stats[L][0].append(g), stats[L][1].append(r)
                if L in keep:
                    h = engram.apply(L, h, hashes)
            h, pm = blk(h, start_pos, pm, None)
        return head(h, pm)

    t = time.time()
    lg = backbone(prompt, 0, a.variant == "both")
    log(f"prefill {time.time() - t:.0f}s")
    tv, ti, lse, full = [], [], [], []
    for step in range(a.steps):
        # wait for the reference continuation (written every 8 steps by ref_e2e_gen)
        while True:
            try:
                st = torch.load(f"{REF}/results.pt")["stream"]
                if st.shape[1] >= S + step + 2:
                    break
            except Exception:
                pass
            time.sleep(60)
        stream = st[: a.B]
        l = lg.float()
        v, i = l.topk(50, -1)
        tv.append(v), ti.append(i), lse.append(torch.logsumexp(l, -1))
        if step < 48:
            full.append(l.to(torch.bfloat16))
        if step == a.steps - 1:
            break
        lg = backbone(stream[:, S + step : S + step + 1], S + step, a.variant == "both")
        log(f"step {step}")
        out = dict(
            variant=a.variant,
            topk_val=torch.stack(tv),
            topk_idx=torch.stack(ti),
            lse=torch.stack(lse),
            full=torch.stack(full),
        )
        out["stats"] = {L: (torch.cat(g), torch.cat(r)) for L, (g, r) in stats.items() if g}
        torch.save(out, f"{REF}/noeng_{a.variant}.pt")
    log("DONE")


if __name__ == "__main__":
    main()
