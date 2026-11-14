# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dump-diff of layer 0 (or LAYER) of the chain vs the fp32 oracle on the 4 short prompts: every iHC / attention / MLP intermediate. Env LAYER (default 0)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.reference import oracle as O
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER = int(os.environ.get("LAYER", "0"))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ids_all = torch.load("/out/prompt_ids.pt")
    md = open_mesh(trace=500_000_000)
    try:
        ch = Hy4Chain(md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), ckpt="/ckpt")
        ch.add_layers(LAYER, LAYER + 1)
        ch.build_head()
        ly = ch.layers[0]
        ids = torch.full((4, CHUNK), 120002, dtype=torch.long)
        for r in range(4):
            ids[r, : len(ids_all[r])] = torch.tensor(ids_all[r])
        taps = {}
        x4 = ch.embed_ids(ids)
        if LAYER > 0:
            raise SystemExit("LAYER>0 needs the oracle input stream: not supported here")
        xin = [ttnn.to_torch(ttnn.get_device_tensors(x4)[r * 8]).float() for r in range(4)]
        out = ly.prefill_chunk(x4, 0, {}, taps=taps)
        o = O.Hy4Oracle("/ckpt")
        cfg, conv = o.cfg, O.Conv()
        sd = o.ck.layer_state(LAYER)
        for r in range(4):
            S = len(ids_all[r])
            x4o = o.embed_ids(torch.tensor([ids_all[r]]))
            res = {"row": r}
            res["x4_in"] = round(pcc(xin[r].reshape(4, -1, 6144)[:, :S].permute(1, 0, 2), x4o[0]), 6)
            hp = O.hc_params(sd, "attn")
            pre, post = O.hc_pre_post(x4o, hp, cfg, conv)
            raw = O.hc_collapse(x4o, pre)
            h = O.rms_norm(raw, sd["input_layernorm.weight"], cfg.norm_eps)
            a = O.attn_block(h, raw, sd, cfg, conv, None)
            x4a = O.hc_expand(x4o, a, post)
            pre2, post2 = O.hc_pre_post(x4a, O.hc_params(sd, "mlp"), cfg, conv)
            h2 = O.rms_norm(O.hc_collapse(x4a, pre2), sd["post_attention_layernorm.weight"], cfg.norm_eps)
            y = O.dense_mlp(h2, sd, cfg, conv) if cfg.is_dense(LAYER) else None
            ref = {
                "pre_a": pre[0],
                "post_a": post[0],
                "h_a": h[0],
                "a": a[0],
                "x4a": x4a[0],
                "pre_m": pre2[0],
                "post_m": post2[0],
                "h_m": h2[0],
            }
            if y is not None:
                ref["y"] = y[0]
            for k, v in ref.items():
                d = taps[k][r]
                if k in ("pre_a", "post_a", "pre_m", "post_m"):
                    d = d.reshape(4, -1)[:, :S].T  # [S,4]
                elif k == "x4a":
                    d = d.reshape(4, -1, 6144)[:, :S].permute(1, 0, 2)
                else:
                    d = d.reshape(-1, 6144)[:S]
                res[k] = [round(pcc(d, v), 6), round(rel(d, v), 5)]
            print("DBG " + json.dumps(res), flush=True)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
