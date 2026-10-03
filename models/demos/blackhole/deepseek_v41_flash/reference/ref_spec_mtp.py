# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference dump of the DSpark drafter (mtp.0..2) from the saved hidden states of ``ref_spec_accept`` (results.pt): only the 3 stages are built
(~7 GB), so it runs in about a minute. Saves ``--out``/mtp_ref.pt with, for the first ``--steps`` decode steps:

    ring[s]   [B,128,512]  window cache of stage s after the prefill seed (slot = position % 128), before step 0
    steps[t]  {pos, tok (t_{pos+1}), main_hidden [B,15360], stage_out [3][B,5,4,5120] (streams after each stage), logits [B,5,V] (markov-biased),
               conf [B,5], drafts [B,6]}
"""

import argparse
import json
import os

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels as K
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.ref_spec_accept import CKPT, build_mtp
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/mnt/tt-data/ssinghal/dsv4-spec-accept/results.pt")
    ap.add_argument("--out", default="/mnt/tt-data/ssinghal/dsv4-spec-accept/mtp_ref.pt")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(16)
    K.FAKE_QUANT = False
    res = torch.load(a.results)
    mod = R.load_model_module()
    torch.set_default_dtype(torch.bfloat16)
    B = res["prompt"].shape[0]
    args = R.model_args(B, 256)
    args.temperature = 0
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    sh = _Shards()
    stages = build_mtp(mod, args, index, sh.get("embed.weight"), sh.get("head.weight"), B)
    S = res["S"]
    hid_pre = res["main_hidden_pre"].to(torch.bfloat16)
    stream = res["stream"]
    # seed
    h, main_x = stages[0].forward_embed(hid_pre, stream[:, S])
    pre = mod.make_identity_pre_mix(h, 4)
    for st in stages:
        h, pre = st(h, 0, pre, main_x)
    out = {"S": S, "ring": [st.attn.window_kv_cache.clone().float() for st in stages], "steps": []}
    for t in range(a.steps):
        pos = S + t
        tok = stream[:, pos + 1]
        hid = res["main_hidden_dec"][t].to(torch.bfloat16)[:, None]
        h, main_x = stages[0].forward_embed(hid, tok)
        pre = mod.make_identity_pre_mix(h, 4)
        outs = []
        for st in stages:
            h, pre = st(h, pos, pre, main_x)
            outs.append(h.clone().float())
        ids, logits, conf = stages[-1].forward_head(h, pre, tok)
        out["steps"].append(
            {
                "pos": pos,
                "tok": tok,
                "main_hidden": hid[:, 0].float(),
                "stage_out": outs,
                "pre": pre.clone().float(),
                "logits": logits.float(),
                "conf": conf.float(),
                "drafts": ids,
                "saved_drafts": res["drafts"][t],
            }
        )
        print(f"step {t}: drafts match saved: {bool((ids == res['drafts'][t]).all())}", flush=True)
    torch.save(out, a.out)


if __name__ == "__main__":
    main()
