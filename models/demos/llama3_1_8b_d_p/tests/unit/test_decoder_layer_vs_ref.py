# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: one complete decoder layer with residuals vs the torch reference.

The composition test, run after every piece passes alone: input_layernorm -> attention -> residual ->
post_attention_layernorm -> MLP -> residual. Random weights identical on both sides.

Run at (1,1) — pure math, no collectives — and at the (8,4) target, where the sequence is SP-sharded
across rows and every collective in the layer (attention's o_proj all-reduce, the MLP's down_proj
all-reduce) is live.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention import allocate_kv_cache
from models.demos.llama3_1_8b_d_p.tt.layer import DecoderLayer

from ..test_factory import (
    concat_sp,
    dev0,
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    replicate,
    shard_seq_on_sp,
)

PCC = 0.99


def layer_state_dict(ref_layer, head_dim):
    """Reference decoder-layer weights in the substate layout ``DecoderLayer`` expects."""
    a = ref_layer.self_attn
    attn = hf_to_meta_qk(
        {
            "q_proj.weight": a.q_proj.weight.data,
            "k_proj.weight": a.k_proj.weight.data,
            "v_proj.weight": a.v_proj.weight.data,
            "o_proj.weight": a.o_proj.weight.data,
        },
        head_dim,
    )
    sd = {f"self_attn.{k}": v for k, v in attn.items()}
    sd.update(
        {
            "mlp.gate_proj.weight": ref_layer.mlp.gate_proj.weight.data,
            "mlp.up_proj.weight": ref_layer.mlp.up_proj.weight.data,
            "mlp.down_proj.weight": ref_layer.mlp.down_proj.weight.data,
            "input_layernorm.weight": ref_layer.input_layernorm.weight.data,
            "post_attention_layernorm.weight": ref_layer.post_attention_layernorm.weight.data,
        }
    )
    return sd


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [512], ids=["s512"])
def test_decoder_layer_vs_ref_single(mesh_device, device_params, seq_len, reset_seeds):
    """Composition on one chip: no SP, no collectives — isolates the layer math."""
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_layer = ref.LlamaDecoderLayer(cfg, layer_idx=0)
    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    reference, _, _ = ref_layer(x, (cos_hf, sin_hf))

    layer = DecoderLayer(
        mesh_device,
        cfg,
        layer_state_dict(ref_layer, hd),
        layer_idx=0,
        ccl_manager=None,
        mesh_config=mesh_config,
        transformation_mats={"prefill": tt_rope.build_transformation_mat(mesh_device)},
        max_seq_len=seq_len,
        attn_weight_dtype=ttnn.bfloat16,
        mlp_weight_dtype=ttnn.bfloat16,
        sequence_parallel=False,
    )

    cos_m, sin_m = ref.build_cos_sin_meta(seq_len, cfg)
    rope_mats = [replicate(mesh_device, cos_m), replicate(mesh_device, sin_m)]
    x_tt = replicate(mesh_device, x.reshape(1, 1, seq_len, cfg.hidden_size))
    out = dev0(layer(x_tt, position_embeddings=rope_mats)).reshape(1, seq_len, cfg.hidden_size)

    passing, pcc = comp_pcc(reference, out, PCC)
    logger.info(f"decoder layer 1x1 s={seq_len}: {pcc}")
    assert passing, f"PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])
def test_decoder_layer_sp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Composition at the production mesh: SP=8 sequence shard, TP=4, every collective live."""
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    n_kv_local = cfg.num_key_value_heads // mesh_config.tp

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_layer = ref.LlamaDecoderLayer(cfg, layer_idx=0)
    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    reference, _, _ = ref_layer(x, (cos_hf, sin_hf))

    layer = DecoderLayer(
        mesh_device,
        cfg,
        layer_state_dict(ref_layer, hd),
        layer_idx=0,
        ccl_manager=make_ccl(mesh_device),
        mesh_config=mesh_config,
        transformation_mats={"prefill": tt_rope.build_transformation_mat(mesh_device)},
        max_seq_len=seq_len,
        attn_weight_dtype=ttnn.bfloat16,
        mlp_weight_dtype=ttnn.bfloat16,
        sequence_parallel=True,
    )

    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=seq_len, chunk_size=seq_len, sp_axis=mesh_config.sp_axis
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=seq_len,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    x_tt = shard_seq_on_sp(mesh_device, x.reshape(1, 1, seq_len, cfg.hidden_size), mesh_config)
    out_tt = layer(x_tt, position_embeddings=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=0, indexed_rope=True)
    out = concat_sp(mesh_device, out_tt, mesh_config).reshape(1, seq_len, cfg.hidden_size)

    passing, pcc = comp_pcc(reference, out, PCC)
    logger.info(f"decoder layer SP=8xTP=4 s={seq_len}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
