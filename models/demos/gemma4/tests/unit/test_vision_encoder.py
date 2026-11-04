# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

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
from models.demos.gemma4.tests.test_factory import skip_unless_siglip_vision
from models.demos.gemma4.tt.attention.prefill import (
    build_mm_prefill_masks,
    mm_allowed_matrix,
    vision_block_ids,
    vision_block_spans,
    vision_placeholder_ids,
)
from models.demos.gemma4.tt.generator import items_within_prefill_window, merge_soft_tokens
from models.demos.gemma4.tt.vision.embedders import PREFIX, PROJECTION_KEY, Gemma4UnifiedVisionEmbedder
from models.demos.gemma4.tt.vision.vision_attention import VisionAttention
from models.demos.gemma4.tt.vision.vision_block import VisionBlock
from models.demos.gemma4.tt.vision.vision_mlp import Gemma4VisionMLP
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_tower import (
    VisionPatchEmbedder,
    VisionPooler,
    VisionRotaryEmbedding,
    VisionTower,
    VisionTransformer,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    map_hf_to_meta_keys,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 2],  # None means all layers, specific numbers will run fewer layers
    ids=["all_layers", "single_layer", "two_layers"],
)
@pytest.mark.parametrize(
    # image_grid_thw: The temporal, height and width of feature shape of each image in LLM.
    "token_budget, image_grid_chw",
    [
        (
            1120 * 9,
            [3, 110, 85],
        ),  # 300 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2550x3300
        (
            280 * 9,
            [3, 50, 50],
        ),  # 240 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2048x1300
    ],
    ids=["300dpi", "240dpi"],
)
@skip_unless_siglip_vision()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_model_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
    num_layers,
    is_ci_env,
    request,
    token_budget,
    image_grid_chw,
):
    test_id = request.node.callspec.id
    if is_ci_env and "two_layers" not in test_id and os.environ.get("GEMMA4_VISION_FULL_DEPTH") != "1":
        pytest.skip("CI only runs the two_layers test (GEMMA4_VISION_FULL_DEPTH=1 lifts this)")

    dtype = ttnn.bfloat8_b
    # Set from measurement, not from another repo's number. Full depth against the
    # correctly masked reference measures ~0.99, so 0.98 is earned with margin.
    # The previous 0.91 was justified as "Llama 3 repo allows 0.91 for prefill,
    # vision probably even less sensitive to pcc" -- a guess, and it sat below a
    # real defect (no bidirectional padding mask) rather than above a result.
    pcc = 0.99 if num_layers and num_layers <= 3 else 0.98
    batch_size = 1  # For prefill we only support batch_size = 1
    ref_seq_len = image_grid_chw[1] * image_grid_chw[2]
    seq_len = ((token_budget // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.num_hidden_layers = num_layers
        from transformers import logging as transformers_logging

        # Set logging level to ERROR to suppress warnings about unexpected keys
        transformers_logging.set_verbosity_error()
    else:
        num_layers = model_args.hf_config.vision_config.num_hidden_layers

    # Create reference model
    reference_model = model_args.reference_vision_model(depth=model_args.hf_config.vision_config.num_hidden_layers)
    state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
    state_dict = convert_vision_block_hf_to_meta(
        state_dict, model_args.n_heads, model_args.n_kv_heads, model_args.head_dim
    )
    state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}
    print(state_dict.keys())

    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]  # [batch, num_patches, 2]
    pt_pixel_values = processed["pixel_values"]  # [batch, num_patches, in_dim]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, num_patches]
    ref_seq_len = pixel_position_ids.shape[1]
    logger.info(f"num_patches={ref_seq_len}, valid_patches={(~padding_positions).sum().item()}")

    # Reference: patch embed -> encoder. Patch embedding lives in VisionTower now, so the encoder
    # (both reference and TT) is fed already-embedded patch hidden states.
    inputs_embeds = reference_model.patch_embedder(pt_pixel_values, pixel_position_ids, padding_positions)
    reference_output = reference_model.encoder(
        inputs_embeds=inputs_embeds,
        pixel_position_ids=pixel_position_ids,
        # As Gemma4VisionModel does. Previously None, to match a TT encoder that
        # applied no padding mask; that handicap hid the defect (see the tower test).
        attention_mask=~padding_positions,
    ).last_hidden_state

    # Initialize TT model (rotary + transformer blocks, all on device)
    tt_model = VisionTransformer(
        args=model_args,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    # Upload the (host-computed) patch embeddings + position metadata as ttnn tensors. The encoder
    # no longer performs patch embedding, so it is fed already-embedded hidden states directly.
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_inputs_embeds = ttnn.from_torch(
        inputs_embeds.unsqueeze(0),  # [1, batch, num_patches, hidden]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_position_ids = ttnn.from_torch(
        pixel_position_ids.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    # Run TT model
    tt_out = tt_model(
        tt_inputs_embeds,
        tt_position_ids,
        unpadded_seq_len=ref_seq_len,
        seq_len=seq_len,
        # Real patch count, so the encoder applies the same bidirectional padding
        # mask the reference gets above. Note this is a DIFFERENT padding from
        # ``unpadded_seq_len``: that is tile padding (num_patches -> seq_len),
        # this is the (-1,-1) image padding inside num_patches.
        num_valid_patches=int((~padding_positions).sum().item()),
    )

    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )

    tt_output_torch = tt_out[:, 0:1, :, : model_args.hf_config.vision_config.hidden_size].squeeze(0).squeeze(0)

    # Compare the VALID patches only. Padded rows are masked out of attention on
    # both sides, so what they contain is undefined and differs between the two
    # implementations -- and ``Gemma4VisionModel`` discards them anyway via the
    # pooler mask, exactly as the tower test does. Including them would compare
    # values no consumer ever reads.
    valid = ~padding_positions.squeeze(0)  # [num_patches], True = real patch
    ref_valid = reference_output.squeeze()[valid]
    tt_valid = tt_output_torch[valid]
    passing, pcc_message = comp_pcc(ref_valid, tt_valid, pcc)
    logger.info(comp_allclose(ref_valid, tt_valid))
    logger.info(f"PCC of output: {pcc_message}")

    # Generate test summary message
    test_desc = f"Vision Transformer Model ({num_layers} layers)"

    if passing:
        logger.info(f"{test_desc} Passed!")
    else:
        logger.warning(f"{test_desc} Failed! PCC value is lower than {pcc} for some of the outputs. Check Warnings!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


# ===== merged from tests/unit/test_vision_attention.py
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


@skip_unless_siglip_vision()
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
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
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


# ===== merged from tests/unit/test_vision_patch_embedder.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "token_budget",
    (1120 * 9, 560 * 9),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_patch_embedder_inference(token_budget, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=token_budget)
    reference_model = model_args.reference_patch_embedder()  # HF Gemma4VisionPatchEmbedder
    state_dict = reference_model.state_dict()

    # Build patchified pixel values + patch position metadata with the image processor (aspect 4:3).
    scale = int(math.sqrt(token_budget / 12))
    image_grid_chw = [3, scale * 3, scale * 4]
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_values = processed["pixel_values"]  # [batch, num_patches, 3*patch_size^2]
    pixel_position_ids = processed["image_position_ids"]  # [batch, num_patches, 2]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, num_patches]
    logger.info(f"pixel_values={tuple(pixel_values.shape)}, num_patches={pixel_position_ids.shape[1]}")

    reference_output = reference_model(pixel_values, pixel_position_ids, padding_positions)

    tt_model = VisionPatchEmbedder(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_cache_path=None,  # Don't cache random weights
        dtype=dtype,
    )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tt_input = ttnn.from_torch(
        pixel_values.unsqueeze(0),  # [1, batch, num_patches, in_dim]
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    # pixel_position_ids / padding_positions are on-device ttnn tensors (int32 / uint8, row-major).
    tt_position_ids = ttnn.from_torch(
        pixel_position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    tt_padding_positions = ttnn.from_torch(
        padding_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    logger.info("Run VisionPatchEmbedder")
    tt_output = tt_model(tt_input, pixel_position_ids=tt_position_ids, padding_positions=tt_padding_positions)

    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch[0]  # [1, batch, num_patches, hidden] -> [batch, num_patches, hidden]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("VisionPatchEmbedder Passed!")
    else:
        logger.warning("VisionPatchEmbedder Failed!")

    assert passing, f"VisionPatchEmbedder output does not meet PCC requirement {pcc_required}: {pcc_message}."


# ===== merged from tests/unit/test_vision_rotary_embedding.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "token_budget",
    (1120 * 9, 560 * 9),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_rotary_embedding_inference(token_budget, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=token_budget)
    reference_model = model_args.reference_rotary_emb()  # HF Gemma4VisionRotaryEmbedding

    # Build patch position metadata with the image processor (aspect ratio 4:3).
    scale = int(math.sqrt(token_budget / 12))
    image_grid_chw = [3, scale * 3, scale * 4]
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]  # [batch, num_patches, 2]
    logger.info(f"num_patches={pixel_position_ids.shape[1]}")

    # Reference rotary embedding (x only provides dtype/device), converted to the Meta interleaved
    # layout that the TT module (and on-device attention) uses.
    dummy_x = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    cos_ref, sin_ref = reference_model(dummy_x, pixel_position_ids)
    cos_ref, sin_ref = convert_rope_style_hf_to_meta_md(cos_ref, sin_ref)

    tt_model = VisionRotaryEmbedding(mesh_device=mesh_device, args=model_args, dtype=dtype)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    # position_ids enters as a working-dtype (bf16) ttnn tensor, no typecast inside the module.
    tt_position_ids = ttnn.from_torch(
        pixel_position_ids.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    logger.info("Run VisionRotaryEmbedding")
    tt_cos, tt_sin = tt_model(tt_position_ids)

    def to_torch(t):
        out = ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if is_mesh else ttnn.to_torch(t)
        return out[
            0, :, :, : model_args.head_dim
        ]  # [1, batch, num_patches, head_dim] -> [batch, num_patches, head_dim]

    tt_cos_torch = to_torch(tt_cos)
    tt_sin_torch = to_torch(tt_sin)

    cos_passing, cos_msg = comp_pcc(cos_ref, tt_cos_torch, pcc_required)
    sin_passing, sin_msg = comp_pcc(sin_ref, tt_sin_torch, pcc_required)

    logger.info(comp_allclose(cos_ref, tt_cos_torch))
    logger.info(f"cos PCC: {cos_msg}")
    logger.info(f"sin PCC: {sin_msg}")
    if cos_passing and sin_passing:
        logger.info("VisionRotaryEmbedding Passed!")
    else:
        logger.warning("VisionRotaryEmbedding Failed!")

    assert cos_passing, f"cos PCC too low: {cos_msg}"
    assert sin_passing, f"sin PCC too low: {sin_msg}"


# ===== merged from tests/unit/test_vision_mlp.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "rows",
    (
        2048,  # TODO: fix padding issues
        4096,
        2048 * 5
        # 14308, # from 3B test image
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mlp_inference(rows, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = Mode.PREFILL  # Vision processing is prefill only (generating token embeddings)

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=rows)
    reference_model = model_args.reference_mlp()
    state_dict = convert_hf_to_meta(reference_model.state_dict(), model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("Gemma4VisionMLP", 0)
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    tt_ccl = TT_CCL(mesh_device)
    tt_model = Gemma4VisionMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=None,  # Don't cache random weights
        layer_num=0,
        # dtype=dtype,
        # model_config=model_args.get_model_config()
    )
    torch_input = torch.randn(1, 1, rows, model_args.hf_config.vision_config.hidden_size, dtype=torch.bfloat16)
    reference_output = reference_model(torch_input)
    # MLP receives replicated input; gate/up are column-parallel and down is row-parallel,
    # so the all-reduce inside the MLP leaves the full hidden dim on every device.
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run MLP")
    tt_output = tt_model(tt_input, mode)

    # Output is replicated after the all-reduce: concat the per-device copies on dim 1
    # and keep the first.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )

    tt_output_torch = tt_output_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("MLP Passed!")
    else:
        logger.warning("MLP Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."


# ===== merged from tests/unit/test_vision_block.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_block_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    n_layers = 27
    dtype = ttnn.bfloat16
    pccs = [0.97] * n_layers
    pccs[24:] = [0.85] * (n_layers - 24)
    print(pccs)
    batch_size = 1  # For prefill we only support batch_size = 1

    # for this test assume 1 image of size 84 x 112
    image_grid_chw = [3, 30, 40]
    ref_seq_len = 1120 * 9
    # pad seq_len to be divisible by base_model_args.MAX_QKV_MM_SEQ_LEN from the tt_transformers model
    seq_len = ((ref_seq_len // 2048) + 1) * 2048  # Using 128 as MAX_QKV_MM_SEQ_LEN

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    reference_whole_model = model_args.reference_vision_model()

    all_passing = True
    for layer_num in range(n_layers):
        logger.info(f"Testing layer {layer_num}")
        pcc = pccs[layer_num]

        reference_model = reference_whole_model.encoder.layers[layer_num]

        # Get the state dict of the reference model.
        # Use the vision-specific block converter: it applies the per-block (2D) RoPE
        # permute to the attention q/k weights and norms and renames only the projection
        # submodules, preserving the HF container/norm names (self_attn, input_layernorm,
        # post_attention_layernorm, ...) that the Gemma-4 vision modules read. The stock
        # convert_hf_to_meta would (a) skip the q/k permute and (b) apply LLM renames
        # (self_attn->attention, input_layernorm->attention_norm), breaking key lookups.
        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_vision_block_hf_to_meta(
            state_dict,
            model_args.n_heads,
            model_args.n_kv_heads,
            model_args.head_dim,
        )
        state_dict_prefix = model_args.get_state_dict_prefix("VisionBlock", layer_num)
        state_dict = {f"{state_dict_prefix}{k}": v for k, v in state_dict.items()}
        print(state_dict.keys())

        # Example inputs and preprocessing
        pt_input = torch.randn(1, 1, ref_seq_len, model_args.dim, dtype=torch.bfloat16)
        random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
        img = T.ToPILImage()(random_img)

        print("model_args.head_dim", model_args.head_dim)
        image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
        processed = image_processor(images=[img], max_soft_tokens=1120, return_tensors="pt")
        pixel_position_ids = processed["image_position_ids"]
        position_embeddings = model_args.reference_vision_model().encoder.rotary_emb(pt_input, pixel_position_ids)

        # pre-compute the rotational embedding matrix and send to device
        cos, sin = position_embeddings
        cos, sin = convert_rope_style_hf_to_meta_md(cos, sin)
        cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0)
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

        # Initialize TT model
        tt_model = VisionBlock(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=None,  # Don't cache random weights
            layer_num=layer_num,
            tt_ccl=TT_CCL(mesh_device),
            dtype=dtype,
            transformation_mats=transformation_mats,
            args=model_args,
        )

        # Prepare input tensor for the TT model
        tt_input = pt_input.clone()
        tt_input = torch.nn.functional.pad(tt_input, (0, 0, 0, seq_len - ref_seq_len))
        attention_input = model_args.prepare_residual_tensor_prefill(tt_input.view(1, seq_len, -1))

        # Run our model
        print(cos.shape, sin.shape)
        tt_out = tt_model(
            attention_input,
            rot_mats=rot_mats,
        )

        # Process the output
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
        )
        tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)  # [batch, seq, hidden_dim]

        # Remove sequence padding
        tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

        # Run reference model
        print(pt_input.shape, position_embeddings[0].shape, pixel_position_ids.shape)
        reference_output = reference_model(
            pt_input.squeeze(0),
            position_embeddings=position_embeddings,
            position_ids=pixel_position_ids,
        )

        # Compare outputs
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"Vision Block Layer {layer_num} Passed!")
        else:
            logger.warning(f"Vision Block Layer {layer_num} Failed! PCC: {pcc_message} lower than {pcc}")
            all_passing = False

    assert all_passing, "PCC value is lower than expected for some of the outputs. Check Layer-specific Warnings!"


# ===== merged from tests/unit/test_vision_pooler.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "token_budget",
    (1120 * 9, 560 * 9),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_pooler_inference(token_budget, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=token_budget)
    reference_model = model_args.reference_pooler()  # HF Gemma4VisionPooler
    hidden_size = model_args.hf_config.vision_config.hidden_size
    pooling_kernel_size = model_args.hf_config.vision_config.pooling_kernel_size

    # Build patch position metadata with the image processor (aspect ratio 4:3), exactly as
    # the vision attention/encoder tests do.
    scale = int(math.sqrt(token_budget / 12))
    image_grid_chw = [3, scale * 3, scale * 4]
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]  # [batch, seq, 2]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, seq]

    seq_len = pixel_position_ids.shape[1]
    output_length = seq_len // (pooling_kernel_size**2)
    logger.info(f"seq_len={seq_len}, output_length={output_length}")

    # Random encoder output features (the pooler is independent of how they were produced).
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    reference_output, reference_mask = reference_model(
        hidden_states=hidden_states,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=output_length,
    )

    tt_model = VisionPooler(mesh_device=mesh_device, args=model_args, dtype=dtype)

    # Pooler expects [1, batch, seq, hidden] replicated across the mesh.
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    tt_input = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    logger.info("Run VisionPooler")
    tt_output, tt_mask = tt_model(
        tt_input,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=output_length,
    )

    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)
    # [1, batch, output_length, hidden] -> [batch, output_length, hidden]
    tt_output_torch = tt_output_torch[0]

    # The validity mask is pure metadata and must match the reference exactly.
    assert torch.equal(tt_mask, reference_mask), "Pooler validity mask does not match reference."

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("VisionPooler Passed!")
    else:
        logger.warning("VisionPooler Failed!")

    assert passing, f"VisionPooler output does not meet PCC requirement {pcc_required}: {pcc_message}."


@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_pooler_no_pooling_keeps_the_valid_patches(mesh_device, reset_seeds, ensure_gc):
    """``output_length == seq_len`` skips the pooling matmul: the padding patches are zeroed and the
    returned mask marks the VALID patches (True), as in the pooled branch and as the callers strip."""
    dtype = ttnn.bfloat16
    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=560 * 9)
    hidden_size = model_args.hf_config.vision_config.hidden_size

    seq_len, n_pad = 64, 16
    hidden_states = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    idx = torch.arange(seq_len)
    pixel_position_ids = torch.stack([idx // 8, idx % 8], dim=-1).unsqueeze(0)  # [1, seq, 2]
    padding_positions = torch.zeros(1, seq_len, dtype=torch.bool)
    padding_positions[0, -n_pad:] = True
    pixel_position_ids[padding_positions] = -1

    tt_model = VisionPooler(mesh_device=mesh_device, args=model_args, dtype=dtype)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    tt_input = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output, tt_mask = tt_model(
        tt_input,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=seq_len,
    )

    assert torch.equal(tt_mask, torch.logical_not(padding_positions)), "no-pooling mask must mark the valid patches"
    out = (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))[0]
    expected = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0).float() * math.sqrt(hidden_size)
    assert torch.all(out[0, -n_pad:] == 0), "padding patches must be zeroed"
    passing, pcc_message = comp_pcc(expected, out.float(), 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"no-pooling output does not match the reference: {pcc_message}"


# ===== merged from tests/unit/test_vision_tower.py
@skip_unless_siglip_vision()
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 2],
    ids=["all_layers", "single_layer", "two_layers"],
)
@pytest.mark.parametrize(
    "token_budget, image_grid_chw",
    [
        (1120 * 9, [3, 110, 85]),  # 300 DPI scanned Letter-size doc (~2550x3300)
        (560 * 9, [3, 66, 54]),  # 240 DPI scanned Letter-size doc
        # (140 * 9, [3, 34, 27])
    ],
    ids=["300dpi", "240dpi"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_tower_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
    num_layers,
    is_ci_env,
    request,
    token_budget,
    image_grid_chw,
):
    test_id = request.node.callspec.id
    if is_ci_env and "two_layers" not in test_id and os.environ.get("GEMMA4_VISION_FULL_DEPTH") != "1":
        pytest.skip("CI only runs the two_layers test (GEMMA4_VISION_FULL_DEPTH=1 lifts this)")

    dtype = ttnn.bfloat8_b
    # Full depth measured 0.9887 (300dpi) / 0.9912 (240dpi) against the correctly
    # masked reference, so 0.98 is earned with margin. It was 0.91 while the
    # encoder applied no padding mask, where full depth scored 0.9246 / 0.8569 --
    # a threshold set below a known defect rather than above a measured result.
    pcc = 0.99 if num_layers and num_layers <= 3 else 0.98
    batch_size = 1  # prefill only supports batch_size = 1
    seq_len = ((token_budget // 2048) + 1) * 2048  # pad to a multiple of MAX_QKV_MM_SEQ_LEN (2048)

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.num_hidden_layers = num_layers
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    else:
        num_layers = model_args.hf_config.vision_config.num_hidden_layers

    # Reference: the full HF vision tower (patch embed -> encoder -> pooler).
    reference_model = model_args.reference_vision_model(depth=model_args.hf_config.vision_config.num_hidden_layers)

    state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
    state_dict = convert_vision_block_hf_to_meta(
        state_dict, model_args.n_heads, model_args.n_kv_heads, model_args.head_dim
    )
    state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    # Build patch position metadata + pixels with the image processor (aspect ratio 4:3).
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)

    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]  # [batch, num_patches, 2]
    pt_pixel_values = processed["pixel_values"]  # [batch, num_patches, in_dim]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, num_patches]
    num_patches = pixel_position_ids.shape[1]
    pooling_kernel_size = model_args.hf_config.vision_config.pooling_kernel_size
    output_length = num_patches // (pooling_kernel_size**2)
    logger.info(f"num_patches={num_patches}, output_length={output_length}")

    # Reference: compose the HF submodules exactly as ``Gemma4VisionModel.forward``
    # does, including ``attention_mask=~padding_positions``. This previously passed
    # ``attention_mask=None`` to match a TT encoder that applied no padding mask --
    # i.e. the reference was handicapped to match the implementation, which hid the
    # defect and cost ~0.03 PCC of apparent quality. The mask is implemented now.
    inputs_embeds = reference_model.patch_embedder(pt_pixel_values, pixel_position_ids, padding_positions)
    encoder_output = reference_model.encoder(
        inputs_embeds=inputs_embeds, attention_mask=~padding_positions, pixel_position_ids=pixel_position_ids
    ).last_hidden_state
    pooled_ref, pooler_mask = reference_model.pooler(
        hidden_states=encoder_output,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=output_length,
    )
    reference_output = pooled_ref[pooler_mask]  # [num_valid, hidden]
    if model_args.hf_config.vision_config.standardize:
        reference_output = (reference_output - reference_model.std_bias) * reference_model.std_scale

    # TT vision tower.
    tt_model = VisionTower(
        args=model_args,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    tt_pooled, tt_mask = tt_model(pt_pixel_values, pixel_position_ids, seq_len)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    tt_pooled_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled)[0]) if is_mesh else ttnn.to_torch(tt_pooled)
    tt_pooled_torch = tt_pooled_torch[0]  # [1, batch, output_length, hidden] -> [batch, output_length, hidden]

    # Strip padded soft tokens the same way the reference does (hidden_states[pooler_mask]).
    tt_output_torch = tt_pooled_torch[tt_mask]  # [num_valid, hidden]

    logger.info(f"reference={tuple(reference_output.shape)}, tt={tuple(tt_output_torch.shape)}")
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC of output: {pcc_message}")

    test_desc = f"Vision Tower ({num_layers} layers)"
    if passing:
        logger.info(f"{test_desc} Passed!")
    else:
        logger.warning(f"{test_desc} Failed! PCC value is lower than {pcc}. Check Warnings!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


# ===== merged from tests/unit/test_unified_vision_embedder.py: On-device Gemma-4 unified (encoder-free) vision embedder vs the HF reference.
def _hf_embedder():
    from transformers import AutoConfig

    try:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedVisionEmbedder as HFVisionEmbedder,
        )
    except ImportError:  # transformers without the unified Gemma 4 classes
        pytest.skip("transformers has no gemma4_unified module")

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-12B-it")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(getattr(cfg, "vision_config", None), "model_patch_size", None) is None:
        # Tower-style (SigLIP) config, e.g. the 31B or the checked-in config-only
        # dirs: the encoder-free embedder does not apply.
        pytest.skip(f"{model_path} is not a unified (encoder-free) Gemma 4 checkpoint")
    torch.manual_seed(0)
    vis = HFVisionEmbedder(cfg.vision_config, cfg.text_config)
    # Random init: the checkpoint's constructor leaves the position table at zero
    # and LayerNorms at identity, which would not exercise the lookup / affine paths.
    with torch.no_grad():
        for p in vis.parameters():
            p.normal_(0.0, 0.05)
    assert isinstance(vis.multimodal_embedder, Gemma4UnifiedMultimodalEmbedder)
    return vis, cfg


def _state_dict_from_hf(vis):
    sd = {}
    for k, v in vis.state_dict().items():
        if k.startswith("multimodal_embedder.embedding_projection."):
            sd[PROJECTION_KEY] = v.detach().clone()
        else:
            sd[PREFIX + k] = v.detach().clone()
    return sd


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_valid, num_patches", [(280, 280), (150, 288)], ids=["full", "padded"])
def test_unified_vision_embedder(mesh_device, num_valid, num_patches):
    vis, cfg = _hf_embedder()
    patch_dim = cfg.vision_config.model_patch_size**2 * 3
    torch.manual_seed(1)
    pixel_values = torch.randn(1, num_patches, patch_dim)
    side = int(cfg.vision_config.mm_posemb_size)
    pos = torch.stack(
        [torch.randint(0, side, (num_patches,)), torch.randint(0, side, (num_patches,))], dim=-1
    ).unsqueeze(0)
    pos[0, num_valid:] = -1  # padding patches

    with torch.no_grad():
        ref = vis(pixel_values.to(torch.bfloat16), pos).float()[0]  # [P, text_hidden]

    tt = Gemma4UnifiedVisionEmbedder(
        mesh_device,
        _state_dict_from_hf(vis),
        text_hidden_size=cfg.text_config.hidden_size,
    )
    out_tt, mask = tt.forward(pixel_values, pos)
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None
    )
    out = out[0, 0, :num_patches].float() if out.dim() == 4 else out[:num_patches].float()

    assert mask.shape == (1, out_tt.shape[2])
    assert int(mask.sum()) == num_valid
    passing, pcc = comp_pcc(ref[:num_valid], out[:num_valid], 0.99)
    logger.info(f"unified vision embedder PCC over {num_valid} valid patches: {pcc}")
    assert passing, f"PCC {pcc} < 0.99"


# ===== merged from tests/unit/test_multimodal_merge.py: Host-side check of the multimodal soft-token merge (no device): image, video and
IMG, VID, AUD, PAD = 258880, 258884, 258881, 0


def _prompt(spec):
    """spec: list of ('t', n) text ids or (token_id, n) placeholder runs -> ids tensor."""
    ids = []
    for kind, n in spec:
        ids += [1000 + len(ids) + i for i in range(n)] if kind == "t" else [kind] * n
    return torch.tensor(ids, dtype=torch.long)


def _embeds(n, tag):
    # row r of modality `tag` carries (tag, r) so the merged order can be read back
    e = torch.zeros(n, 4)
    e[:, 0] = tag
    e[:, 1] = torch.arange(n)
    return e


def test_merge_orders_modalities_by_prompt_position():
    ids = _prompt([("t", 3), (AUD, 5), ("t", 2), (IMG, 4), ("t", 1), (VID, 6), ("t", 2)])
    llm_ids, positions, merged = merge_soft_tokens(
        ids, [(IMG, _embeds(4, 1)), (VID, _embeds(6, 2)), (AUD, _embeds(5, 3))], PAD
    )
    # every placeholder became pad, text untouched
    assert (
        (llm_ids[ids == IMG] == PAD).all() and (llm_ids[ids == VID] == PAD).all() and (llm_ids[ids == AUD] == PAD).all()
    )
    text = (ids != IMG) & (ids != VID) & (ids != AUD)
    assert torch.equal(llm_ids[text], ids[text])
    # positions ascending and exactly the placeholder slots
    expected = torch.where((ids == IMG) | (ids == VID) | (ids == AUD))[0]
    assert positions.dtype == torch.int32 and torch.equal(positions.long(), expected)
    # the soft tokens follow the prompt: audio (tag 3) first, then image (1), then video (2)
    tags = merged[:, 0].tolist()
    assert tags == [3] * 5 + [1] * 4 + [2] * 6
    # and within a modality the rows keep their own order
    assert merged[:5, 1].tolist() == list(range(5)) and merged[5:9, 1].tolist() == list(range(4))
    assert merged.dtype == torch.bfloat16


def test_merge_single_modality_and_two_images():
    ids = _prompt([("t", 2), (IMG, 3), ("t", 1), (IMG, 3)])
    llm_ids, positions, merged = merge_soft_tokens(ids.view(1, -1), [(IMG, _embeds(6, 1))], PAD)
    assert positions.tolist() == [2, 3, 4, 6, 7, 8]
    assert merged[:, 1].tolist() == list(range(6))  # both images' tokens, in order


def test_merge_rejects_count_mismatch_and_missing_id(expect_error):
    ids = _prompt([("t", 2), (AUD, 4)])
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(AUD, _embeds(5, 3))], PAD)
    with expect_error(ValueError, "no placeholder id"):
        merge_soft_tokens(ids, [(None, _embeds(4, 3))], PAD)


def test_merge_cached_prefix_fills_trailing_placeholders(expect_error):
    # vLLM prefix caching: the first image sits inside the cached prefix and arrives
    # without its payload, so only the second image's 3 soft tokens are forwarded.
    ids = _prompt([("t", 2), (IMG, 3), ("t", 1), (IMG, 3), ("t", 2)])
    llm_ids, positions, merged = merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD, num_cached=6)
    assert positions.tolist() == [6, 7, 8] and merged[:, 1].tolist() == [0, 1, 2]
    assert (llm_ids[ids == IMG] == PAD).all()  # every placeholder still becomes pad
    # a shortfall that would leave an uncached placeholder empty is an error ...
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD, num_cached=4)
    # ... and so is a shortfall with no cached prefix at all
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD)


def test_items_past_the_chunk_end_wait_for_a_later_chunk():
    # Chunked prefill under load: the first chunk of a 1200-token prompt ends at 300,
    # before its image (offset 420, 270 placeholders); the plugin still forwards the
    # image, so it must be left out of this chunk's merge and taken by the next.
    assert items_within_prefill_window([420], [270], 300) == []
    assert items_within_prefill_window([420], [270], 1200) == [0]
    # Several items: the third lies past the end, the first two (one of them inside a
    # cached prefix) are inside.
    assert items_within_prefill_window([10, 420, 900], [270, 270, 256], 700) == [0, 1]
    # The window never ends inside an item (vLLM disable_chunked_mm_input); an exact end
    # is inside.
    assert items_within_prefill_window([10], [270], 280) == [0]
    # Older plugins without ranges: every item is kept, as before.
    assert items_within_prefill_window(None, None, 300) == []
    assert items_within_prefill_window([None, 420], [None, 270], 300) == [0]


# ===== merged from tests/unit/test_mm_attention_mask.py: Host test: the image-span prefill masks match HF's block-bidirectional causal / sliding masks.
IMG_mm_attention_mask, VID_mm_attention_mask, AUD_mm_attention_mask, TXT = 1000, 1001, 1002, 7


def _prompt_mm_attention_mask(spans):
    """spans: list of (token, count) -> flat ids"""
    ids = []
    for tok, n in spans:
        ids += [tok] * n
    return torch.tensor(ids, dtype=torch.long)


@pytest.mark.parametrize(
    "spans",
    [
        [(TXT, 5), (IMG_mm_attention_mask, 6), (TXT, 4)],
        [
            (TXT, 3),
            (IMG_mm_attention_mask, 4),
            (TXT, 1),
            (IMG_mm_attention_mask, 4),
            (TXT, 2),
        ],  # two images -> two blocks
        [
            (IMG_mm_attention_mask, 4),
            (VID_mm_attention_mask, 3),
            (TXT, 2),
        ],  # adjacent image+video placeholders form ONE run (as HF)
        [(TXT, 2), (AUD_mm_attention_mask, 5), (TXT, 2)],  # audio stays causal: no block
        [(TXT, 9)],
    ],
)
def test_block_ids_match_hf(spans):
    from transformers.models.gemma4.modeling_gemma4 import get_block_sequence_ids_for_mask

    ids = _prompt_mm_attention_mask(spans)
    # HF token types: 1 image, 2 video, 3 audio, 0 text
    types = torch.zeros_like(ids)
    types[ids == IMG_mm_attention_mask] = 1
    types[ids == VID_mm_attention_mask] = 2
    types[ids == AUD_mm_attention_mask] = 3
    ref = get_block_sequence_ids_for_mask(types.view(1, -1), device="cpu").view(-1)
    got = vision_block_ids(ids, (IMG_mm_attention_mask, VID_mm_attention_mask))
    assert torch.equal(got, ref), (got, ref)


@pytest.mark.parametrize("sliding_window", [None, 4])
def test_allowed_matrix_matches_hf_mask_functions(sliding_window):
    from transformers.masking_utils import (
        blockwise_overlay,
        causal_mask_function,
        or_masks,
        sliding_window_causal_mask_function,
    )

    ids = _prompt_mm_attention_mask(
        [(TXT, 3), (IMG_mm_attention_mask, 6), (TXT, 2), (IMG_mm_attention_mask, 3), (TXT, 2)]
    )
    blocks = vision_block_ids(ids, IMG_mm_attention_mask)
    S = int(ids.shape[0])
    base = causal_mask_function if sliding_window is None else sliding_window_causal_mask_function(sliding_window)
    fn = or_masks(base, blockwise_overlay(blocks.view(1, -1)))
    ref = torch.zeros(S, S, dtype=torch.bool)
    for q in range(S):
        for k in range(S):
            ref[q, k] = bool(fn(0, 0, torch.tensor(q), torch.tensor(k)))
    got = mm_allowed_matrix(blocks, S, sliding_window)
    assert torch.equal(got, ref)
    # an image block wider than the window still attends to itself fully
    if sliding_window is not None:
        assert got[8, 3] and not got[8, 0]


def test_additive_masks_and_padding():
    ids = _prompt_mm_attention_mask([(TXT, 2), (IMG_mm_attention_mask, 4), (TXT, 1)])
    blocks = vision_block_ids(ids, IMG_mm_attention_mask)
    masks = build_mm_prefill_masks(blocks, seq_len=16, sliding_window=4)
    assert set(masks) == {"full_attention", "sliding_attention"}
    full, sliding = masks["full_attention"], masks["sliding_attention"]
    assert full.shape == (1, 1, 16, 16) and full.dtype == torch.bfloat16
    assert float(full[0, 0, 2, 5]) == 0.0  # image token 2 sees image token 5 (future, same block)
    assert float(full[0, 0, 1, 5]) < -1e8  # text before the image does not see it (bf16-rounded MASK_NEG)
    assert float(sliding[0, 0, 15, 10]) < -1e8 and float(sliding[0, 0, 15, 12]) == 0.0  # window on padding rows
    # every row keeps at least its own position
    assert bool((full[0, 0].diagonal() == 0).all())
    assert (
        build_mm_prefill_masks(vision_block_ids(_prompt_mm_attention_mask([(TXT, 5)]), IMG_mm_attention_mask), 8, 4)
        is None
    )


def test_block_spans_key():
    ids = _prompt_mm_attention_mask(
        [(TXT, 2), (IMG_mm_attention_mask, 3), (TXT, 1), (IMG_mm_attention_mask, 2), (TXT, 1)]
    )
    assert vision_block_spans(vision_block_ids(ids, IMG_mm_attention_mask)) == ((2, 5), (6, 8))
    assert vision_block_spans(vision_block_ids(_prompt_mm_attention_mask([(TXT, 4)]), IMG_mm_attention_mask)) == ()


def test_placeholder_ids_cover_video_blocks():
    """The generator must pass BOTH placeholder ids: with the image id alone a video-only prefill stays causal."""
    assert vision_placeholder_ids(IMG_mm_attention_mask) == [IMG_mm_attention_mask]
    assert vision_placeholder_ids(IMG_mm_attention_mask, None) == [IMG_mm_attention_mask]
    assert vision_placeholder_ids(IMG_mm_attention_mask, VID_mm_attention_mask) == [
        IMG_mm_attention_mask,
        VID_mm_attention_mask,
    ]
    assert vision_placeholder_ids(IMG_mm_attention_mask, IMG_mm_attention_mask) == [IMG_mm_attention_mask]
    ids = _prompt_mm_attention_mask([(TXT, 3), (VID_mm_attention_mask, 5), (TXT, 2)])
    assert (
        int(vision_block_ids(ids, vision_placeholder_ids(IMG_mm_attention_mask)).max()) < 0
    )  # the bug: video block lost
    blocks = vision_block_ids(ids, vision_placeholder_ids(IMG_mm_attention_mask, VID_mm_attention_mask))
    assert blocks.tolist() == [-1, -1, -1, 0, 0, 0, 0, 0, -1, -1]
    assert build_mm_prefill_masks(blocks, 16, sliding_window=4) is not None
