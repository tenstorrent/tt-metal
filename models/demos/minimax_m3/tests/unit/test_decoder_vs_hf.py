# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-2 end-to-end PCC test: full MiniMax-M2 decoder layer vs the HF reference
MiniMaxM2DecoderLayer.

This is the first test that exercises the WHOLE layer wired together:
  RMSNorm -> attention (QKV/qk-norm/partial-RoPE/SDPA/o_proj) -> residual ->
  RMSNorm -> MoE (sigmoid+bias router -> SiLU-SwiGLU experts) -> residual.

Reduced expert count (256 -> 32) so the HF reference fits in host RAM and the run
is fast; bf8 expert weights to isolate wiring/logic from bfp4 quantization.
Runs at mesh (1,1)/TP=1/EP=1.

MiniMax-M2's modeling code ships WITH the checkpoint (not vendored). Set HF_MODEL to a
downloaded MiniMax-M2 checkpoint to run this; otherwise it skips.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.layer import DecoderLayer
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import hf_model_path, parametrize_mesh_with_fabric, requires_hf_reference


@requires_hf_reference
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_decoder_layer_prefill_vs_hf(mesh_device, device_params, seq_len, reset_seeds):
    ref_path = hf_model_path()
    config = AutoConfig.from_pretrained(ref_path, trust_remote_code=True)
    config._attn_implementation = "eager"
    # Reduce experts so the HF reference (a ModuleList of MLPs) fits in host RAM.
    config.num_local_experts = 32
    config.num_experts_per_tok = 8
    H = config.hidden_size

    DecoderRef = get_class_from_dynamic_module("modeling_minimax_m2.MiniMaxM2DecoderLayer", ref_path)
    RotRef = get_class_from_dynamic_module("modeling_minimax_m2.MiniMaxM2RotaryEmbedding", ref_path)

    # --- HF reference layer ---
    ref_layer = DecoderRef(config, layer_idx=0).eval()
    rotary = RotRef(config)
    x = torch.randn(1, seq_len, H)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        ref_out = ref_layer(hidden_states=x, position_embeddings=(cos, sin), attention_mask=causal_mask)
    ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out

    # --- TT decoder layer from the same weights ---
    # Attention q/k weights + q/k-norm need the partial-rotary Meta swizzle; the
    # helper matches the *.q_proj.weight / *.k_proj.weight / *.q_norm.weight /
    # *.k_norm.weight suffixes regardless of the 'self_attn.' prefix.
    state = convert_hf_qkv_to_meta_format_partial(ref_layer.state_dict(), config.head_dim, config.rotary_dim)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=config, datatype=ttnn.bfloat16)
    trans_mats = rope_setup.get_both_trans_mats()

    decoder = DecoderLayer(
        mesh_device,
        config,
        state,
        layer_idx=0,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        transformation_mats=trans_mats,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=1,
        use_throughput_experts=False,
        expert_weight_dtype=ttnn.bfloat8_b,  # isolate wiring from bfp4 quantization
    )

    rope_mats = [
        rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
        rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
    ]
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, H),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = decoder(x_tt, position_embeddings=rope_mats)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, seq_len, H)

    passing, pcc = comp_pcc(ref_out, out, 0.97)
    logger.info(f"decoder layer prefill vs HF (E=32): {pcc}")
    assert passing, f"decoder layer PCC fail: {pcc}"
