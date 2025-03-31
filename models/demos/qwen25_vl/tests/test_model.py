# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.qwen25_vl.tt.model import VisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.tt_transformers.tt.common import (
    get_rot_transformation_mat,
    PagedAttentionConfig,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull

# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from models.demos.qwen25_vl.reference.model import Qwen2_5_VisionTransformerPretrainedModel
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys


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
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 3],  # None means all layers, specific numbers will run fewer layers
    ids=["all_layers", "single_layer", "three_layers"],
)
def test_vision_model_inference(
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    num_layers,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    mesh_device.enable_async(False)

    # Example inputs for http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
    # pixel_values are produced by Qwen2_5_VLImageProcessor, these come from the above img
    pt_pixel_values = torch.randn([14308, 1176])
    # image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #     The temporal, height and width of feature shape of each image in LLM.
    # for this test assume 1 image of size 98 x 146 patches as used in with their repo example img
    image_grid_thw = torch.tensor([[1, 98, 146]])
    ref_seq_len = image_grid_thw[0, 1] * image_grid_thw[0, 2]
    # pad seq_len to be divisible by 128 (MAX_QKV_MM_SEQ_LEN from tt_transformers model)
    seq_len = ((ref_seq_len // 128) + 1) * 128

    model_args = VisionModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.depth = num_layers
    else:
        num_layers = model_args.hf_config.vision_config.depth

    # Create reference model
    reference_model = Qwen2_5_VisionTransformerPretrainedModel(model_args.hf_config.vision_config)
    # FIXME: state_dict = model_args.load_state_dict()
    state_dict = standardize_hf_keys(reference_model.state_dict())
    state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    # Set up paged attention config
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
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Get the necessary preprocessing for vision model
    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=model_args.head_dim,
        spatial_merge_size=model_args.hf_config.vision_config.spatial_merge_size,
        window_size=model_args.hf_config.vision_config.window_size,
        patch_size=model_args.hf_config.vision_config.patch_size,
    )

    # Pre-compute the rotational embedding matrix and send to device
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

    # Preprocess patch embeddings for the TT model in torch for now
    patch_input = reference_model.patch_embed(pt_pixel_values)
    patch_seq_len, _ = patch_input.shape
    spatial_merge_unit = model_args.hf_config.vision_config.spatial_merge_size**2
    patch_input = patch_input.reshape(patch_seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    patch_input = patch_input[window_index, :, :]
    patch_input = patch_input.reshape(patch_seq_len, -1)

    # Initialize TT model
    tt_model = VisionTransformer(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=None,  # NOCOMMIT model_args.weight_cache_path(dtype),
        dtype=dtype,
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )

    # Prepare input tensor for the TT model
    tt_input = torch.nn.functional.pad(patch_input, (0, 0, 0, seq_len - ref_seq_len))
    tt_input = model_args.prepare_residual_tensor_prefill(
        tt_input,
        force_replicated=False if model_args.is_galaxy else True,
    )

    # Run TT model (only blocks, not patch embedding/merging)
    tt_out = tt_model(
        tt_input,
        unpadded_seq_len=ref_seq_len,
        cu_seqlens=cu_seqlens,
        cu_window_seqlens=cu_window_seqlens,
        rot_mats=rot_mats,
        user_id=0,
        page_table=page_table_tt,
    )

    # Run reference model
    reference_output = reference_model(pt_pixel_values, image_grid_thw)

    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.hf_config.vision_config.out_hidden_size].squeeze(0).squeeze(0)

    # Post-process in torch
    tt_output_torch = tt_output_torch[torch.argsort(window_index), :]

    # Compare outputs
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    # Generate test summary message
    test_desc = f"Vision Transformer Model ({num_layers} layers)"

    if passing:
        logger.info(f"{test_desc} Passed!")
    else:
        logger.warning(f"{test_desc} Failed!")
    assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
