# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""forward_decode PCC: incremental single-token MLA decode with on-device KV cache vs prefill golden.

Decodes the reference sequence one token at a time through TtMistral4MLA.forward_decode (q/k/v for
the token -> update_cache at cur_pos -> scaled_dot_product_attention_decode -> o_proj), feeding the
reference per-position RoPE. The stacked decode outputs must match the prefill golden mla_out — i.e.
the on-device KV cache + decode op reproduce the full-attention result.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4MLA


def _repl(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _pos(p, B, mesh):
    return ttnn.from_torch(
        torch.tensor([p] * B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_forward_decode(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, 1, 0, 32)
    mla_in, mla_out = g["mla_in"], g["mla_out"]
    cos, sin = g["rope_cos"], g["rope_sin"]
    B, S, rope = mla_in.shape[0], mla_in.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)
    kv = mla.init_kv_cache(B, max_seq=64)

    outs = []
    for i in range(S):
        x_i = _repl(mla_in[:, i : i + 1, :], mesh_device)
        c_i = _repl(cos[:, i : i + 1, :].reshape(B, 1, 1, rope), mesh_device)
        s_i = _repl(sin[:, i : i + 1, :].reshape(B, 1, 1, rope), mesh_device)
        o = mla.forward_decode(x_i, _pos(i, B, mesh_device), c_i, s_i, kv)
        outs.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B])
    out = torch.cat(outs, dim=1)  # [B, S, hidden]
    passing, msg = comp_pcc(mla_out, out, pcc_required)
    logger.info(f"MLA forward_decode (KV-cache incremental) PCC: {msg}")
    assert passing, f"forward_decode PCC below {pcc_required}: {msg}"
