# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-layer (untiled by TP) Qwen3.6-27B weights for the pipeline-parallel prefill (see pp_prefill.py).

The (1 + w) RMSNorm weights are folded into the rows of the projection that follows them, as for the
resident decode (weights.py); column orders:
  GDN in-projection   q (16 x 128) | k (16 x 128) | v (48 x 128) | z (48 x 128) | a (48) | b (48)
  attention in-proj   q (24 x 256) | gate (24 x 256) | k (4 x 256) | v (4 x 256)
  MLP                 gate, up, down
The GDN output norm's weight (per head dim) is folded into the out-projection rows.
"""
import torch

from qwen36_resident.weights import _fold
from qwen36_resident.model import DV, HD, NQ, NV, is_attn

VD = NV * DV


def gdn_layer(ck, i):
    ln = ck.layer(i, "input_layernorm.weight")
    qkv = ck.layer(i, "linear_attn.in_proj_qkv.weight")  # [2 KD + VD, H]: q | k | v
    z = ck.layer(i, "linear_attn.in_proj_z.weight")
    a = ck.layer(i, "linear_attn.in_proj_a.weight")
    b = ck.layer(i, "linear_attn.in_proj_b.weight")
    rows = torch.cat([qkv, z, a, b])
    return dict(
        W=_fold(ln, rows.T),
        taps=ck.layer(i, "linear_attn.conv1d.weight")[:, 0].float().bfloat16().float(),  # [conv_ch, 4]
        dt=ck.layer(i, "linear_attn.dt_bias").float(),
        neg_a=-ck.layer(i, "linear_attn.A_log").float().exp(),
        # the gated norm's weight is per head dim: folded into the out-projection rows of every head
        out=(
            ck.layer(i, "linear_attn.norm.weight").float().bfloat16().float().repeat(VD // DV)[:, None]
            * ck.layer(i, "linear_attn.out_proj.weight").T.float()
        ).bfloat16(),
    )


def attn_layer(ck, i):
    ln = ck.layer(i, "input_layernorm.weight")
    qg = ck.layer(i, "self_attn.q_proj.weight").reshape(NQ, 2, HD, -1)
    # q | k | v | gate: q, k, v are one contiguous slice of the projection
    rows = torch.cat(
        [
            qg[:, 0].reshape(NQ * HD, -1),
            ck.layer(i, "self_attn.k_proj.weight"),
            ck.layer(i, "self_attn.v_proj.weight"),
            qg[:, 1].reshape(NQ * HD, -1),
        ]
    )
    return dict(
        W=_fold(ln, rows.T),
        wq=(1.0 + ck.layer(i, "self_attn.q_norm.weight").float()).bfloat16().float(),
        wk=(1.0 + ck.layer(i, "self_attn.k_norm.weight").float()).bfloat16().float(),
        out=ck.layer(i, "self_attn.o_proj.weight").T.contiguous(),
    )


def mlp(ck, i):
    ln = ck.layer(i, "post_attention_layernorm.weight")
    g, u, dn = (ck.layer(i, f"mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
    return dict(G=_fold(ln, g.T), U=_fold(ln, u.T), D=dn.T.contiguous())


def layer(ck, i, interval):
    mixer = attn_layer(ck, i) if is_attn(i, interval) else gdn_layer(ck, i)
    return dict(mixer=mixer, mlp=mlp(ck, i))
