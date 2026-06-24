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
from models.demos.gemma4.tests.unit.test_vision_attention import convert_rope_style_hf_to_meta_md
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_rotary_embedding import VisionRotaryEmbedding


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
