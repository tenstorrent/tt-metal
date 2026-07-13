# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-2 module-by-module PCC tests for MiniMax-M2:
build the HuggingFace reference module + the TT module from the SAME random weights
and compare outputs by PCC.

MiniMax-M2 is a trust_remote_code model: its modeling code ships WITH the checkpoint
(not vendored into this repo). Set HF_MODEL to a downloaded MiniMax-M2 checkpoint to
run these; otherwise they skip. (transformers==4.57.1 — see requirements.txt.)

These instantiate the full HF config (62 layers' worth of dims) but build only a
single attention module, so they run at mesh (1,1) / TP=1 on a single card. The
TP>1 / CCL paths need a multi-card system.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m2.config import MeshConfig, ModeConfig
from models.demos.minimax_m2.tt.attention import Attention, AttentionConfig
from models.demos.minimax_m2.tt.attention_configs import MiniMaxM2AttentionProgramConfig
from models.demos.minimax_m2.tt.ccl import CCLManager
from models.demos.minimax_m2.tt.model import create_rope_setup
from models.demos.minimax_m2.utils.general_utils import get_default_num_links
from models.demos.minimax_m2.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import hf_model_path, parametrize_mesh_with_fabric, requires_hf_reference


@requires_hf_reference
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 8)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_attention_prefill_vs_hf(mesh_device, device_params, seq_len, reset_seeds):
    """Full attention block (QKV -> qk-norm -> partial RoPE -> SDPA -> o_proj) vs HF reference."""
    ref_path = hf_model_path()
    config = AutoConfig.from_pretrained(ref_path, trust_remote_code=True)
    config._attn_implementation = "eager"
    hidden = config.hidden_size

    AttnRef = get_class_from_dynamic_module("modeling_minimax_m2.MiniMaxM2Attention", ref_path)
    RotRef = get_class_from_dynamic_module("modeling_minimax_m2.MiniMaxM2RotaryEmbedding", ref_path)

    # --- HF reference ---
    ref_attn = AttnRef(config, layer_idx=0).eval()
    rotary = RotRef(config)
    x = torch.randn(1, seq_len, hidden)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_out, _ = ref_attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=causal_mask)

    # --- TT module from the same weights (partial-rotary Meta RoPE swizzle) ---
    # Uses the SAME production helper the real weight-load path uses.
    state = convert_hf_qkv_to_meta_format_partial(ref_attn.state_dict(), config.head_dim, config.rotary_dim)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    # Linear topology: this Galaxy is a plain MESH (no torus). Harmless at TP=1.
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=config, datatype=ttnn.bfloat16)
    trans_mats = rope_setup.get_both_trans_mats()

    attn_config = AttentionConfig(
        hidden_size=hidden,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rotary_dim=config.rotary_dim,
        rms_norm_eps=config.rms_norm_eps,
        use_qk_norm=config.use_qk_norm,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=1,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=MiniMaxM2AttentionProgramConfig(),
        layer_idx=0,
        transformation_mats=trans_mats,
        create_kv_cache=True,
    )

    rope_mats = [
        rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
        rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
    ]
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, hidden),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = attn(x_tt, rope_mats=rope_mats, position_idx=None, page_table=None, kv_cache=None)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(1, seq_len, hidden)

    passing, pcc = comp_pcc(ref_out, out, 0.97)
    logger.info(f"attention prefill vs HF: {pcc}")
    assert passing, f"attention PCC fail: {pcc}"
