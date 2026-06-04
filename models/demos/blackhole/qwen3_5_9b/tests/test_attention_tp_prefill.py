# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 validation for Qwen3.5 full-attention PREFILL (causal) on a (1,4) mesh.

Full causal GQA torch reference over a short sequence covers the prefill head
layout (transpose/reshape), partial RoPE over positions, causal SDPA, gate, and
the row-parallel output + reduce-scatter.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=/home/ttuser/atupe/Qwen27b \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_attention_tp_prefill.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tests.test_attention_tp import _load_attn_layer, _mp
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs


def _rope_torch(x, rope_dim, theta):  # x: [S, H, HD]
    S = x.shape[0]
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    emb = torch.cat([torch.outer(torch.arange(S).float(), inv)] * 2, dim=-1)  # [S, rope_dim]
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    r1, r2 = xr[..., : rope_dim // 2], xr[..., rope_dim // 2 :]
    xrot = torch.cat([-r2, r1], dim=-1)
    return torch.cat([xr * cos + xrot * sin, xp], dim=-1)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_tp_prefill(mesh_device, reset_seeds, ensure_gc):
    mp = _mp()
    os.environ.setdefault("HF_MODEL", mp)
    S = 64
    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    logger.info(f"devices={nd} full-attn layer={li} S={S}")

    sd = _load_attn_layer(mp, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, S, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    out = attn.forward_prefill(x_tt, cos, sin)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0))[0, 0].float()

    # ---- torch reference: causal GQA attention ----
    NH, NKV, HD = args.n_heads, args.n_kv_heads, args.head_dim
    grp, rd, scale = NH // NKV, args.rope_head_dim, HD**-0.5
    xf = x[0, 0].float()
    qg = (xf @ sd["q_proj.weight"].float().T).reshape(S, NH, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:]
    k = (xf @ sd["k_proj.weight"].float().T).reshape(S, NKV, HD)
    v = (xf @ sd["v_proj.weight"].float().T).reshape(S, NKV, HD)
    qn = sd["q_norm.weight"].float() + 1.0
    kn = sd["k_norm.weight"].float() + 1.0
    q = q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * qn
    k = k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * kn
    q = _rope_torch(q, rd, args.rope_theta)
    k = _rope_torch(k, rd, args.rope_theta)
    k = k[:, torch.arange(NH) // grp, :]  # expand to NH
    v = v[:, torch.arange(NH) // grp, :]
    # scores [NH,S,S]
    qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    cmask = torch.full((S, S), float("-inf")).triu(1)
    attn_w = torch.softmax(scores + cmask, dim=-1)
    ao = torch.matmul(attn_w, vh).permute(1, 0, 2)  # [S,NH,HD]
    gated = ao * torch.sigmoid(gate)
    ref = gated.reshape(S, NH * HD) @ sd["o_proj.weight"].float().T  # [S, dim]

    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(ref, out_t, 0.95)
    logger.info(f"ATTENTION TP PREFILL PCC (S={S}) = {pcc}")
    assert passing, f"attention TP prefill PCC too low: {pcc}"
