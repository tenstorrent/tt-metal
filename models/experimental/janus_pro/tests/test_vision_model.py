# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.janus_pro_vision_model import TtJanusProTransformerVision
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("bsz", [1])
def test_janus_vision_model(
    mesh_device,
    dummy_weights,
    reset_seeds,
    bsz,
):
    pcc_required = 0.95
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device, dummy_weights=dummy_weights)
    state_dict = model_args.load_state_dict()

    # Wrapper composes "model.vision_model." and "model.aligner." internally.
    state_dict_prefix = "model."

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    # HF reference: aligner(vision_model(pixel_values).last_hidden_state) -> (B, 576, 4096).
    # Depending on the transformers version, get_image_features returns either the aligned
    # features tensor directly or a BaseModelOutputWithPooling whose pooler_output holds them.
    reference_model = model_args.reference_vision_transformer(wrap=False)
    reference_model.eval()
    with torch.no_grad():
        image_features = reference_model.model.get_image_features(input_tensor)
    reference_output = getattr(image_features, "pooler_output", image_features)

    tt_model = TtJanusProTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        dtype=dtype,
        configuration=model_args,
    )
    tt_output = tt_model(input_tensor)

    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, :, :, :]

    assert (
        tt_output_torch.shape == reference_output.shape
    ), f"Shape mismatch: tt {tuple(tt_output_torch.shape)} vs ref {tuple(reference_output.shape)}"

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
