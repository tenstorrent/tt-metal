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
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_patch_embedder import VisionPatchEmbedder


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
