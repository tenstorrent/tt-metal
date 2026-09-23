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
from models.demos.gemma4.tt.vision.vision_pooler import VisionPooler


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
