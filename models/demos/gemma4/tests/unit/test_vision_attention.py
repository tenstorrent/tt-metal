# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tt.vision.vision_attention import VisionAttention
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs, vision_mesh_shape_from_env
from models.demos.gemma4.tt.vision.vision_weight_convert import (  # noqa: F401
    convert_rope_style_hf_to_meta_md,
    convert_vision_attention_hf_to_meta,
    convert_vision_block_hf_to_meta,
    meta_permute_norm_weight,
    meta_permute_qk_weight,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [vision_mesh_shape_from_env()],
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
    # Attention consumes a replicated input (the wrapping DistributedNorm produces this).
    attention_input = ttnn.from_torch(
        tt_attention_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(
        attention_input,
        rot_mats=rot_mats,
    )
    # Output is fractured along dim=3 (hidden); concat across TP to recover full dim.
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
        ),
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)

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
