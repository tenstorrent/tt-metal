# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-path PCC for MLA: the decode SDPA op on MLA shapes vs the golden last-token output.

Step 1 of the decode path (decouple the op from KV-cache plumbing): feed the reference's post-RoPE
q/k/v (from get_cached_golden), run ttnn.transformer.scaled_dot_product_attention_decode on the last
query position attending to the full cached k/v, then o_proj, and PCC vs mla_out[last]. Confirms the
decode op reproduces the per-position result for MLA (qk_head_dim 128, v_head 128) before adding the
on-device KV cache (paged_update_cache) + forward_decode.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights


def _repl(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_mla_decode(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, 1, 0, 32)  # 1-layer goldens incl q_states/k_states/value_states, mla_out
    H, qk = cfg.num_attention_heads, cfg.qk_head_dim
    q, k, v = g["q_states"], g["k_states"], g["value_states"]  # [B, H, S, dh]
    B, S = q.shape[0], q.shape[2]
    last = S - 1

    # decode op: q for the last position [1, b, nh, dh]; k/v full sequence [b, nkv, s, dh]; cur_pos=[last]
    q_dec = _repl(q[:, :, last : last + 1, :].permute(2, 0, 1, 3).contiguous(), mesh_device)  # [1,B,H,dh]
    k_tt, v_tt = _repl(k, mesh_device), _repl(v, mesh_device)
    cur = ttnn.from_torch(
        torch.tensor([last] * B, dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q_dec, k_tt, v_tt, cur_pos_tensor=cur, scale=qk ** (-0.5)
    )
    # [1,B,H,dh] -> [B,1,H*dh]
    attn = ttnn.reshape(ttnn.permute(attn, (1, 0, 2, 3)), (B, 1, H * qk))

    tsd = load_m4_weights(ckpt, 1)
    from models.demos.mistral4.tt.mistral4_text import _lin

    o = ttnn.linear(attn, _lin(tsd["model.layers.0.self_attn.o_proj.weight"], mesh_device))
    o_t = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]
    passing, msg = comp_pcc(g["mla_out"][:, last : last + 1, :], o_t, pcc_required)
    logger.info(f"MLA decode-step (op) PCC: {msg}")
    assert passing, f"MLA decode PCC below {pcc_required}: {msg}"
