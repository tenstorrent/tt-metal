# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tests.unit.test_vision_attention import convert_vision_block_hf_to_meta
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_tower import VisionTower
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal


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
    if is_ci_env and "two_layers" not in test_id:
        pytest.skip("CI only runs the two_layers test")

    dtype = ttnn.bfloat8_b
    pcc = 0.99 if num_layers and num_layers <= 3 else 0.91
    batch_size = 1  # prefill only supports batch_size = 1
    seq_len = ((token_budget // 128) + 1) * 128

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

    # Reference: compose the HF submodules with attention_mask=None to match the TT encoder, which
    # does not apply the bidirectional padding mask (same as the encoder unit test). Then pool and
    # strip padded soft tokens exactly as Gemma4VisionModel does.
    inputs_embeds = reference_model.patch_embedder(pt_pixel_values, pixel_position_ids, padding_positions)
    encoder_output = reference_model.encoder(
        inputs_embeds=inputs_embeds, attention_mask=None, pixel_position_ids=pixel_position_ids
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
