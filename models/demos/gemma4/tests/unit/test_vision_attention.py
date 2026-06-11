# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import os

import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tt.vision.vision_attention import VisionAttention
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys, standardize_hf_keys_multimodal
from models.tt_transformers.tt.model_config import ModelArgs


# ---------------------------------------------------------------------------
# Block-aware Meta-RoPE conversion helpers.
#
# Gemma-4 vision uses *multidimensional* (2D) RoPE: head_dim is split into
# `ndim` independent blocks (rows / columns), and standard 1D RoPE is applied
# to each block separately. The model applies `ttnn.experimental.rotary_embedding_llama`
# (Meta interleaved convention) per block, so the weights, q/k norms, and cos/sin
# must be converted to Meta format *per block* rather than across the full head_dim.
# The shared `convert_hf_to_meta` / `convert_rope_style_hf_to_meta` helpers only
# handle 1D RoPE and would mix/destroy the two spatial dimensions.
# ---------------------------------------------------------------------------
def meta_permute_qk_weight(weight, n_heads, head_dim, ndim=2):
    """HF->Meta interleave for q/k projection weights, applied independently per RoPE block."""
    blk = head_dim // ndim
    in_dim = weight.shape[-1]
    weight = weight.view(n_heads, ndim, 2, blk // 2, in_dim)
    weight = weight.transpose(2, 3)
    return weight.reshape(n_heads * head_dim, in_dim)


def meta_permute_norm_weight(weight, head_dim, ndim=2):
    """HF->Meta interleave for q/k RMSNorm weights, applied independently per RoPE block."""
    blk = head_dim // ndim
    weight = weight.view(ndim, 2, blk // 2)
    weight = weight.transpose(1, 2)
    return weight.reshape(head_dim)


def convert_rope_style_hf_to_meta_md(cos, sin, ndim=2):
    """Convert HF cos/sin (per-block half-duplicated) to Meta pairwise-duplicated, per RoPE block."""
    blk = cos.shape[-1] // ndim
    cos_parts, sin_parts = [], []
    for k in range(ndim):
        c = cos[..., k * blk : (k + 1) * blk]
        s = sin[..., k * blk : (k + 1) * blk]
        cos_parts.append(torch.repeat_interleave(c[..., : blk // 2], 2, dim=-1))
        sin_parts.append(torch.repeat_interleave(s[..., : blk // 2], 2, dim=-1))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def convert_vision_attention_hf_to_meta(state_dict, n_heads, n_kv_heads, head_dim, ndim=2):
    """Block-aware HF->Meta conversion for the Gemma-4 vision self-attention weights."""
    converted = {}
    for key, tensor in state_dict.items():
        if "q_proj.linear.weight" in key:
            converted[key] = meta_permute_qk_weight(tensor, n_heads, head_dim, ndim)
        elif "k_proj.linear.weight" in key:
            converted[key] = meta_permute_qk_weight(tensor, n_kv_heads, head_dim, ndim)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted[key] = meta_permute_norm_weight(tensor, head_dim, ndim)
        else:
            converted[key] = tensor
    return map_hf_to_meta_keys(converted)


# Rename only the projection submodules to the names the Gemma-4 vision modules read.
# Crucially this does NOT rename the attention container (self_attn) or the layernorms,
# unlike the stock LLM `map_hf_to_meta_keys` (which renames self_attn->attention,
# input_layernorm->attention_norm, post_attention_layernorm->ffn_norm).
_VISION_BLOCK_PROJ_RENAMES = [
    (".q_proj.", ".wq."),
    (".k_proj.", ".wk."),
    (".v_proj.", ".wv."),
    (".o_proj.", ".wo."),
    (".gate_proj.", ".w1."),
    (".up_proj.", ".w3."),
    (".down_proj.", ".w2."),
]


def convert_vision_block_hf_to_meta(state_dict, n_heads, n_kv_heads, head_dim, ndim=2):
    """Block-aware HF->Meta conversion for a full Gemma-4 vision encoder layer.

    Applies the per-block RoPE permute to the attention q/k weights and q/k norms,
    renames only the projection submodules (q/k/v/o_proj -> wq/wk/wv/wo and
    gate/up/down_proj -> w1/w3/w2), and preserves the HF container/norm key names
    (self_attn, input_layernorm, post_attention_layernorm, pre/post_feedforward_layernorm)
    that the Gemma-4 vision modules expect.
    """
    converted = {}
    for key, tensor in state_dict.items():
        if "q_proj.linear.weight" in key:
            tensor = meta_permute_qk_weight(tensor, n_heads, head_dim, ndim)
        elif "k_proj.linear.weight" in key:
            tensor = meta_permute_qk_weight(tensor, n_kv_heads, head_dim, ndim)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            tensor = meta_permute_norm_weight(tensor, head_dim, ndim)
        new_key = key
        for old, new in _VISION_BLOCK_PROJ_RENAMES:
            new_key = new_key.replace(old, new)
        converted[new_key] = tensor
    return converted


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("token_budget", [140 * 9, 280 * 9, 560 * 9, 1120 * 9])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
def test_vision_attention_inference(
    mesh_device,
    token_budget,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat16  # NOCOMMIT
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    # calculating max patch grid for specified token_budget at aspect ratio 4:3
    scale = int(math.sqrt(token_budget / 12))
    image_grid_chw = [3, scale * 3, scale * 4]
    ref_seq_len = token_budget
    # pad seq_len to be divisible by base_model_args.MAX_QKV_MM_SEQ_LEN
    seq_len = ((ref_seq_len // ModelArgs.MAX_QKV_MM_SEQ_LEN) + 1) * ModelArgs.MAX_QKV_MM_SEQ_LEN

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    reference_model = model_args.reference_attention()

    state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
    state_dict = convert_vision_attention_hf_to_meta(
        state_dict, model_args.n_heads, model_args.n_kv_heads, model_args.head_dim
    )
    state_dict_prefix = model_args.get_state_dict_prefix("VisionAttention", 0)
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    # Example inputs and preprocessing
    pt_attention_input = torch.randn(1, 1, ref_seq_len, model_args.dim, dtype=torch.bfloat16)
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]
    position_embeddings = model_args.reference_vision_model().encoder.rotary_emb(pt_attention_input, pixel_position_ids)

    # pre-compute the rotational embedding matrix and send to device
    cos, sin = position_embeddings

    # Convert HF multidimensional (2D) RoPE cos/sin to Meta pairwise format, per RoPE block
    cos, sin = convert_rope_style_hf_to_meta_md(cos, sin)

    # pad sequence length with cos = 1, sin = 0 (identity rotation)
    cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len), value=1).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len), value=0).unsqueeze(0)
    cos = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = [cos, sin]

    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)

    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}
    tt_ccl = TT_CCL(mesh_device)

    tt_model = VisionAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        tt_ccl=tt_ccl,
        weight_cache_path=None,  # Don't cache random weights
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
    )

    tt_attention_input = pt_attention_input.clone()
    tt_attention_input = torch.nn.functional.pad(tt_attention_input, (0, 0, 0, seq_len - ref_seq_len))
    attention_input = model_args.prepare_residual_tensor_prefill(tt_attention_input.view(1, seq_len, -1))

    tt_out = tt_model(
        attention_input,
        rot_mats=rot_mats,
    )
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)  # [ batch, seq, hidden_dim]

    # Remove sequence padding
    tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

    reference_output = reference_model(
        pt_attention_input.squeeze(0), position_embeddings=position_embeddings, position_ids=pixel_position_ids
    )[0]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Attention Passed!")
    else:
        logger.warning(f"Attention Failed!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
