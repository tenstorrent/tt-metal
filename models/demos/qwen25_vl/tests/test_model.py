# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.demos.qwen25_vl.tt.model import VisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)


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
@pytest.mark.parametrize(
    "num_layers",
    [None, 1, 2],  # None means all layers, specific numbers will run fewer layers
    ids=["all_layers", "single_layer", "two_layers"],
)
@pytest.mark.parametrize(
    # image_grid_thw: The temporal, height and width of feature shape of each image in LLM.
    "seq_len, image_grid_thw",
    [
        (
            42952,
            torch.tensor([[1, 236, 182]]),
        ),  # 300 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2550x3300
        (
            14308,
            torch.tensor([[1, 98, 146]]),
        ),  # 240 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2048x1300
    ],
    ids=["300dpi", "240dpi"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_model_inference(
    mesh_device,
    reset_seeds,
    ensure_gc,
    num_layers,
    is_ci_env,
    request,
    seq_len,
    image_grid_thw,
):
    test_id = request.node.callspec.id
    if is_ci_env and "two_layers" not in test_id:
        pytest.skip("CI only runs the two_layers test")

    dtype = ttnn.bfloat8_b
    pcc = (
        0.99 if num_layers and num_layers <= 3 else 0.91
    )  # Llama 3 repo allows 0.91 for prefill, vision probably even less sensitive to pcc
    batch_size = 1  # For prefill we only support batch_size = 1

    # Example inputs for http://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
    # pixel_values are produced by Qwen2_5_VLImageProcessor, these come from the above img
    pt_pixel_values = torch.randn([seq_len, 1176]) * 0.8320 + 1.2969  # std and mean from above img
    ref_seq_len = image_grid_thw[0, 1] * image_grid_thw[0, 2]
    seq_len = ((ref_seq_len // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
    if num_layers:
        model_args.hf_config.vision_config.depth = num_layers
        from transformers import logging as transformers_logging

        # Set logging level to ERROR to suppress warnings about unexpected keys
        transformers_logging.set_verbosity_error()
    else:
        num_layers = model_args.hf_config.vision_config.depth

    # Create reference model
    reference_model = model_args.reference_vision_model(depth=model_args.hf_config.vision_config.depth)
    # reference_model = Qwen2_5_VisionTransformerPretrainedModel(model_args.hf_config.vision_config)
    # reference_model.load_state_dict(model_args.reference_vision_model().state_dict(), strict=False)
    # FIXME: state_dict = model_args.load_state_dict()
    state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
    state_dict = convert_hf_to_meta(state_dict, model_args.vision_head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    # Initialize TT model
    tt_model = VisionTransformer(
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    # Get the necessary preprocessing for vision model
    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=model_args.vision_head_dim,
        spatial_merge_size=model_args.hf_config.vision_config.spatial_merge_size,
        window_size=model_args.hf_config.vision_config.window_size,
        patch_size=model_args.hf_config.vision_config.patch_size,
    )

    # Pre-compute the rotational embedding matrix and send to device
    cos, sin = position_embeddings
    # thanks, gemini 2.5 pro
    cos, sin = convert_rope_style_hf_to_meta(cos, sin)

    # pad sequence length with cos = 1, sin = 0 (identity rotation)
    cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len), value=1).unsqueeze(0).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len), value=0).unsqueeze(0).unsqueeze(0)
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

    # Initialize TT model
    tt_model = VisionTransformer(
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
    )

    # Prepare input tensor for the TT model
    patch_input = reference_model.patch_embed(pt_pixel_values)  # Use ref model for conv3d for now
    tt_input = tt_model.prepare_input(patch_input, window_index)

    # Run TT model
    tt_out = tt_model(
        tt_input,
        unpadded_seq_len=ref_seq_len,
        cu_seqlens=ttnn.from_torch(cu_seqlens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device),
        cu_window_seqlens=ttnn.from_torch(
            cu_window_seqlens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        ),
        rot_mats=rot_mats,
    )

    # Run reference model
    reference_output = reference_model(pt_pixel_values, image_grid_thw)

    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
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
