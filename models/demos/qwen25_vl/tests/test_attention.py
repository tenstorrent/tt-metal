# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.model_config import ModelArgs
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    PagedAttentionConfig,
)
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta


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
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        #        False,
    ),
    ids=(
        "paged_attention",
        #        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
def test_attention_inference(
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    mesh_device.enable_async(False)

    # Example inputs
    # image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #     The temporal, height and width of feature shape of each image in LLM.
    # for this test assume 1 image of size 98 x 146 as used in their repo as an example
    image_grid_thw = torch.tensor([[1, 98, 146]])
    ref_seq_len = image_grid_thw[0, 1] * image_grid_thw[0, 2]
    # pad seq_len to be divisible by base_model_args.MAX_QKV_MM_SEQ_LEN
    seq_len = ((ref_seq_len // ModelArgs.MAX_QKV_MM_SEQ_LEN) + 1) * ModelArgs.MAX_QKV_MM_SEQ_LEN

    base_model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len)
    vision_model_args = VisionModelArgs(base_model_args)
    reference_model = Qwen2_5_VLVisionAttention(
        base_model_args.hf_config.vision_config.hidden_size, num_heads=base_model_args.hf_config.vision_config.num_heads
    )
    state_dict = vision_model_args.map_keys_to_hf_format(reference_model.state_dict())
    state_dict = convert_hf_to_meta(state_dict, vision_model_args.head_dim)
    state_dict_prefix = vision_model_args.get_state_dict_prefix("Attention", 0)
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    # Example inputs and preprocessing
    pt_attention_input = torch.randn(1, 1, ref_seq_len, vision_model_args.dim)
    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=vision_model_args.head_dim,
        spatial_merge_size=base_model_args.hf_config.vision_config.spatial_merge_size,
        window_size=base_model_args.hf_config.vision_config.window_size,
        patch_size=base_model_args.hf_config.vision_config.patch_size,
    )

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(
        vision_model_args.head_dim,
        mesh_device,
        seq_len,
        # FIXME: what should these be?
        base_model_args.rope_theta,
        base_model_args.rope_scaling_factor,
        base_model_args.orig_context_len,
    )
    transformation_mat_torch = get_rot_transformation_mat(vision_model_args.head_dim)
    print(f"{transformation_mat_torch.shape=}")
    print(f"{rot_mats[0].shape=}")
    print(f"{rot_mats[1].shape=}")

    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            base_model_args.max_batch_size, paged_attention_config.max_num_blocks // base_model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_model = Attention(
        mesh_device,
        state_dict,
        weight_cache_path=base_model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=vision_model_args,
        paged_attention_config=paged_attention_config,
    )

    tt_attention_input = pt_attention_input.clone()
    tt_attention_input = torch.nn.functional.pad(tt_attention_input, (0, 0, 0, seq_len - ref_seq_len))
    print(f"{pt_attention_input.shape=}")
    print(f"{tt_attention_input.shape=}")
    attention_input = base_model_args.prepare_residual_tensor_prefill(
        tt_attention_input,
        force_replicated=False if vision_model_args.is_galaxy else True,
    )

    tt_out = tt_model(
        attention_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=base_model_args.cluster_shape),
    )
    print(f"{tt_out.shape=}")
    tt_output_torch = tt_out[:, 0:1, :, : vision_model_args.dim].view(
        batch_size, seq_len, -1
    )  # [ batch, seq, hidden_dim]
    print(f"{tt_output_torch.shape=}")

    # Remove sequence padding
    tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

    print(f"{position_embeddings[0].shape=}")

    reference_output = reference_model(
        pt_attention_input.squeeze(0).squeeze(0),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=position_embeddings,
    )

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Attention Passed!")
    else:
        logger.warning(f"Attention Failed!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
