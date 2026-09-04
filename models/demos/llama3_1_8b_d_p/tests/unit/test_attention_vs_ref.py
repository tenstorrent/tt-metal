# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the whole attention block vs the torch reference.

QKV proj -> GQA head split -> full RoPE -> causal SDPA -> o_proj -> TP all-reduce, on random weights
identical on both sides, with the SAME cos/sin on both sides so this test measures attention and not
the RoPE constants (those are ``test_rope_vs_ref.py``).

Runs at (1,1) — no collectives, so a failure here is the attention math and nothing else. The TP=4
head split, the fused per-shard QKV interleave and the row-parallel o_proj all-reduce are covered at
the real (8,4) target by ``test_decoder_layer_sp_vs_ref`` / ``test_attention_chunked_vs_ref``; the SP
ring paths by ``test_ring_joint_*``.

A (1,4) submesh case was written and dropped: a plain-MESH Galaxy cannot bring fabric up on a 4-chip
submesh — the routers on the ethernet channels leaving the submesh never complete the remote
handshake, and mesh setup dies with "Fabric Router Sync: Timeout ... (LOCAL_HANDSHAKE_COMPLETE)".
That is a topology limitation of this machine, not of the model, and TP-without-SP is not a
configuration the spec targets (SP=8, TP=4).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig

from ..test_factory import (
    dev0,
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    replicate,
)

PCC = 0.99


def build_reference_attention(cfg, seed=0):
    torch.manual_seed(seed)
    return ref.LlamaAttention(cfg, layer_idx=0)


def attention_state_dict(ref_attn, head_dim):
    """Reference weights in the layout ``load_attention_weights`` expects, Meta-swizzled q/k."""
    sd = {
        "q_proj.weight": ref_attn.q_proj.weight.data,
        "k_proj.weight": ref_attn.k_proj.weight.data,
        "v_proj.weight": ref_attn.v_proj.weight.data,
        "o_proj.weight": ref_attn.o_proj.weight.data,
    }
    return hf_to_meta_qk(sd, head_dim)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128, 512], ids=["s128", "s512"])
def test_attention_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device) if mesh_config.tp > 1 else None

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_attn = build_reference_attention(cfg)
    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    reference, _, _ = ref_attn(x, (cos_hf, sin_hf))

    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            max_seq_len=seq_len,
            sequence_parallel=False,
        ),
        state_dict=attention_state_dict(ref_attn, hd),
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": tt_rope.build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )

    cos_m, sin_m = ref.build_cos_sin_meta(seq_len, cfg)
    rope_mats = [replicate(mesh_device, cos_m), replicate(mesh_device, sin_m)]

    x_tt = replicate(mesh_device, x.reshape(1, 1, seq_len, cfg.hidden_size))
    out = dev0(attn(x_tt, rope_mats=rope_mats)).reshape(1, seq_len, cfg.hidden_size)

    passing, pcc = comp_pcc(reference, out, PCC)
    logger.info(f"attention s={seq_len} mesh={tuple(mesh_device.shape)}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
