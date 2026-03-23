# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test text decoder layer(s) on TG (8,4) to verify numerical correctness.

This fills the gap left by the vision-only unit tests. Run with:
    MESH_DEVICE=TG pytest models/demos/qwen25_vl/tests/test_text_decoder.py -xvs
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("n_layers", [1, 4], ids=["1layer", "4layers"])
@pytest.mark.parametrize("seq_len", [128, 2048], ids=["short", "long"])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_text_decoder_prefill(mesh_device, n_layers, seq_len, reset_seeds, ensure_gc):
    """Run embeddings through N decoder layers on TT and compare to reference."""
    dtype = ttnn.bfloat8_b
    batch_size = 1
    max_seq_len = 4096

    # --- Model args and state dict ---
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
    )
    original_n_layers = model_args.n_layers
    model_args.n_layers = n_layers
    state_dict = model_args.load_state_dict()

    # --- Reference model (HF) ---
    from transformers import AutoModelForImageTextToText

    ref_model_name = model_args.CKPT_DIR  # e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info(f"Loading reference model: {ref_model_name}")
    ref_model = AutoModelForImageTextToText.from_pretrained(ref_model_name, torch_dtype=torch.bfloat16)
    ref_model.eval()

    # --- TT model ---
    logger.info("Creating TT Transformer...")
    tt_model = Transformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    # --- Create input embeddings via reference embed_tokens ---
    input_ids = torch.randint(0, 1000, (1, seq_len))
    embeddings = ref_model.model.language_model.embed_tokens(input_ids)  # [1, seq_len, 3584]

    # --- Compute matching rotary embeddings for both ref and TT ---
    from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

    mrope_position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, -1, -1)  # [3, 1, S]
    rotary_emb = ref_model.model.language_model.rotary_emb
    hf_cos, hf_sin = rotary_emb(embeddings, mrope_position_ids)  # [3, 1, S, head_dim]

    # --- Reference forward through N decoder layers ---
    ref_hidden = embeddings.clone()
    for i in range(n_layers):
        ref_layer = ref_model.model.language_model.layers[i]
        ref_out = ref_layer(
            ref_hidden,
            position_ids=mrope_position_ids,
            position_embeddings=(hf_cos, hf_sin),
        )
        ref_hidden = ref_out[0]

    logger.info(f"Reference output shape: {ref_hidden.shape}")
    logger.info(
        f"Reference stats: mean={ref_hidden.float().mean():.6f} "
        f"std={ref_hidden.float().std():.6f} "
        f"min={ref_hidden.float().min():.4f} max={ref_hidden.float().max():.4f}"
    )

    # --- TT forward (decoder layers only, no LM head) ---
    from models.tt_transformers.tt.common import Mode

    # Replicate the exact cos/sin preparation from multimodal_rope_from_hf:
    # 1. Apply M-RoPE section interleaving (same as apply_multimodal_rotary_pos_emb)
    mrope_section = ref_model.config.rope_scaling["mrope_section"] * 2
    tt_cos = torch.cat([m[i % 3] for i, m in enumerate(hf_cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        1
    )  # [1, 1, S, head_dim]
    tt_sin = torch.cat([m[i % 3] for i, m in enumerate(hf_sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        1
    )  # [1, 1, S, head_dim]
    # 2. Convert HF → meta interleaved format
    tt_cos, tt_sin = convert_rope_style_hf_to_meta(tt_cos, tt_sin)
    rot_mats = (tt_cos, tt_sin)
    prefill_input, tt_rot_mats, tt_page_table, _ = tt_model.prepare_inputs_prefill(embeddings, rot_mats=rot_mats)
    tt_out = tt_model.forward(
        prefill_input,
        current_pos=None,
        rot_mats_global=tt_rot_mats,
        user_id=0,
        mode=Mode.PREFILL,
        get_last_token=-1,
    )

    # --- Extract TT output ---
    # On TG: hidden state is sharded across 4 columns (dim/4=896 each),
    # replicated across 8 rows. Device 0 = row 0, col 0 → first 896 elements.
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    logger.info(f"TT output shape (device 0): {list(tt_out_torch.shape)}")
    logger.info(
        f"TT stats: mean={tt_out_torch.mean():.6f} "
        f"std={tt_out_torch.std():.6f} "
        f"min={tt_out_torch.min():.4f} max={tt_out_torch.max():.4f} "
        f"nan={tt_out_torch.isnan().sum().item()} inf={tt_out_torch.isinf().sum().item()}"
    )

    # --- Compare ---
    # TT output per device: [1, 1, seq_len_padded, dim/cols]
    # Reference: [1, seq_len, dim]
    dim_per_col = model_args.dim // model_args.cluster_shape[1]
    ref_slice = ref_hidden[:, :seq_len, :dim_per_col].unsqueeze(1)  # [1, 1, seq_len, 896]
    tt_slice = tt_out_torch[:, :, :seq_len, :]

    passing, pcc_msg = comp_pcc(ref_slice, tt_slice, 0.90)
    logger.info(f"PCC: {pcc_msg}")

    # Also check that stats are sane
    assert not tt_out_torch.isnan().any(), "TT output contains NaN!"
    assert not tt_out_torch.isinf().any(), "TT output contains Inf!"
    assert tt_out_torch.std() > 1e-6, f"TT output has near-zero std: {tt_out_torch.std()}"

    if passing:
        logger.info(f"Text decoder ({n_layers} layers) PASSED!")
    else:
        logger.warning(f"Text decoder ({n_layers} layers) FAILED!")
    assert passing, f"PCC {pcc_msg} below threshold 0.90"
