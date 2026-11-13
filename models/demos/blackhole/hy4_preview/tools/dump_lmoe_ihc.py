# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""iHC reference dump (layer-moe task, second stage): real iHC streams from the fp32 oracle (read-only ref worktree) and its iHC formulas.
Saves for layer 1: the stream entering layer 1 (x4_in, after layer 0), the attention-side hc params, pre/post, the collapsed + input_layernorm'd sublayer input,
the stream after attention (x4_mid) with the MLP-side pre/post/h, and the expand of a stand-in y (= the real MoE output) onto x4_mid.
    python tools/dump_lmoe_ihc.py OUT_FILE"""
import os
import sys

import torch

REF = "/mnt/tt-data/ssinghal/wt/hy4_ref/models/demos/blackhole/hy4_preview/reference"
import models.demos.blackhole.hy4_preview as _p  # noqa: E402

_p.__path__.append(os.path.dirname(REF))
from models.demos.blackhole.hy4_preview.reference import oracle as O  # noqa: E402
from models.demos.blackhole.hy4_preview.reference.dump_ref import LONG_UNIT, SHORT  # noqa: E402


def main():
    out = sys.argv[1]
    torch.set_num_threads(24)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    mid = "".join(LONG_UNIT.format(i=i, a=(i * 7) % 23 + 1, b=(i * 5) % 17 + 1) for i in range(14))
    ids = [torch.tensor([[120000] + tok.encode(t).ids]) for t in SHORT + [mid]]
    o = O.Hy4Oracle()
    conv = O.Conv()
    cfg = o.cfg
    x4s = [o.embed_ids(i) for i in ids]
    st = [{} for _ in ids]
    x4s = O.decoder_layers_multi(o.ck, 0, x4s, [conv] * len(ids), st, None)
    taps = {}
    x4_out = O.decoder_layers_multi(o.ck, 1, x4s, [conv] * len(ids), st, taps)
    sd = o.ck.layer_state(1)
    res = {
        "x4_in": [],
        "pre_a": [],
        "post_a": [],
        "h_a": [],
        "x4_mid": [],
        "pre_m": [],
        "post_m": [],
        "h_m": [],
        "x4_out": [],
    }
    for p_, x4 in enumerate(x4s):
        pre, post = O.hc_pre_post(x4, O.hc_params(sd, "attn"), cfg, conv)
        raw = O.hc_collapse(x4, pre)
        res["x4_in"].append(x4)
        res["pre_a"].append(pre)
        res["post_a"].append(post)
        res["h_a"].append(O.rms_norm(raw, sd["input_layernorm.weight"], cfg.norm_eps))
        mid_ = taps["after_attn"][p_]
        pre, post = O.hc_pre_post(mid_, O.hc_params(sd, "mlp"), cfg, conv)
        res["x4_mid"].append(mid_)
        res["pre_m"].append(pre)
        res["post_m"].append(post)
        res["h_m"].append(taps["mlp_in"][p_])
        res["x4_out"].append(x4_out[p_])
    res["hc_attn"] = {k: v for k, v in O.hc_params(sd, "attn").items()}
    res["hc_mlp"] = {k: v for k, v in O.hc_params(sd, "mlp").items()}
    res["norm_in"] = sd["input_layernorm.weight"]
    res["norm_post"] = sd["post_attention_layernorm.weight"]
    res["cfg"] = {"hc_eps": cfg.hc_eps, "norm_eps": cfg.norm_eps, "hc_magnitude": cfg.hc_magnitude}
    torch.save(res, out)


if __name__ == "__main__":
    main()
