# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.tt_transformers.tt.common import (
    get_rot_transformation_mat,
)
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys
from models.demos.qwen25_vl.tt.vision_block import VisionBlock


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_num", list(range(32)))
def test_vision_block_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    layer_num,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    mesh_device.enable_async(True)

    # Example inputs
    # image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #     The temporal, height and width of feature shape of each image in LLM.
    # for this test assume 1 image of size 98 x 146 as used in their repo as an example
    image_grid_thw = torch.tensor([[1, 98, 146]])
    ref_seq_len = image_grid_thw[0, 1] * image_grid_thw[0, 2]
    # pad seq_len to be divisible by base_model_args.MAX_QKV_MM_SEQ_LEN from the tt_transformers model
    seq_len = ((ref_seq_len // 128) + 1) * 128  # Using 128 as MAX_QKV_MM_SEQ_LEN

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    reference_model = model_args.reference_vision_block(layer_num)
    # reference_model = Qwen2_5_VLVisionBlock(model_args.hf_config.vision_config)
    # reference_model.load_state_dict(model_args.reference_vision_block(layer_num).state_dict())

    state_dict = standardize_hf_keys(reference_model.state_dict())
    state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("VisionBlock", layer_num)
    state_dict = {f"{state_dict_prefix}{k}": v for k, v in state_dict.items()}

    # Example inputs and preprocessing
    pt_input = torch.randn(1, 1, ref_seq_len, model_args.dim)
    # pt_input = torch.load(f"ref_x_{layer_num - 1}.pt").unsqueeze(0).unsqueeze(0)
    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=model_args.head_dim,
        spatial_merge_size=model_args.hf_config.vision_config.spatial_merge_size,
        window_size=model_args.hf_config.vision_config.window_size,
        patch_size=model_args.hf_config.vision_config.patch_size,
    )

    cu_seqlens = (
        cu_seqlens if layer_num in model_args.hf_config.vision_config.fullatt_block_indexes else cu_window_seqlens
    )

    # pre-compute the rotational embedding matrix and send to device
    cos, sin = position_embeddings
    cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
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
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
    )

    # Prepare input tensor for the TT model
    tt_input = pt_input.clone()
    tt_input = torch.nn.functional.pad(tt_input, (0, 0, 0, seq_len - ref_seq_len))
    tt_input = model_args.prepare_residual_tensor_prefill(
        tt_input.squeeze(0),
        force_replicated=False if model_args.is_galaxy else True,
    )

    # Run our model
    tt_out = tt_model(
        tt_input,
        cu_seqlens=cu_seqlens,
        rot_mats=rot_mats,
    )

    # Process the output
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)  # [batch, seq, hidden_dim]

    # Remove sequence padding
    tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

    # Run reference model
    reference_output = reference_model(
        pt_input.squeeze(0).squeeze(0),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=position_embeddings,
    )

    # Compare outputs
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Vision Block Passed!")
    else:
        logger.warning(f"Vision Block Failed!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
