# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""iHC reference for the dump_v1 streams (ihc-fuse task): the stream entering layers 0..3 (8 prompts x 20 tokens, real fp32 oracle streams), with the oracle
iHC (pre, post, collapse + rms_norm * weight) for BOTH hc parameter sets (attn / mlp) of that layer (read-only oracle in the ref worktree).
    PYTHONPATH=<worktree> python tools/dump_ihc_v1.py OUT_FILE"""
import os
import sys

import torch

REF = "/mnt/tt-data/ssinghal/wt/hy4_ref/models/demos/blackhole/hy4_preview/reference"
import models.demos.blackhole.hy4_preview as _p  # noqa: E402

_p.__path__.append(os.path.dirname(REF))
from models.demos.blackhole.hy4_preview.reference import oracle as O  # noqa: E402

DUMP = "/mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1"


def main():
    out = sys.argv[1]
    torch.set_num_threads(16)
    o = O.Hy4Oracle()
    cfg, conv = o.cfg, O.Conv()
    res = {"cfg": {"hc_eps": cfg.hc_eps, "norm_eps": cfg.norm_eps}, "layers": {}}
    for layer in range(4):
        sd = o.ck.layer_state(layer)
        xs = torch.cat(
            [torch.load(f"{DUMP}/layer{layer:02d}_p{p}.pt", map_location="cpu")["in"] for p in range(8)], dim=1
        )  # [1, 160, 4, D]
        ent = {"x4": xs[0].clone()}
        for which, nw in (("attn", "input_layernorm.weight"), ("mlp", "post_attention_layernorm.weight")):
            hp = O.hc_params(sd, which)
            pre, post = O.hc_pre_post(xs, hp, cfg, conv)
            raw = O.hc_collapse(xs, pre)
            ent[which] = {
                "hc_fn": hp["hc_fn"].float(),
                "hc_scale": hp["hc_scale"].float(),
                "hc_base": hp["hc_base"].float(),
                "norm_w": sd[nw].float(),
                "pre": pre[0],
                "post": post[0],
                "h": O.rms_norm(raw, sd[nw], cfg.norm_eps)[0],
            }
        res["layers"][layer] = ent
        print("layer", layer, "done", flush=True)
    torch.save(res, out)


if __name__ == "__main__":
    main()
