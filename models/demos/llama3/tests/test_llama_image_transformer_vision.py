# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
from models.demos.llama3.tt.llama_image_transformer_vision import TtLlamaCrossAttentionTransformerVision
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(os.environ.get("FAKE_DEVICE"), None)],
    indirect=True,
)
def test_llama_vision_transformer_inference(mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    pcc = 0.82

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    return_intermediate = "3,7,15,23,30"
    return_intermediate = [int(l) for l in return_intermediate.split(",")]

    reference_model = llama_reference_mod.CrossAttentionTransformerVision(model_args)
    reference_model.load_state_dict(partial_state_dict, strict=True)

    all_tests_pass = True

    tt_model = TtLlamaCrossAttentionTransformerVision(
        mesh_device,
        state_dict,
        first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        return_intermediate=return_intermediate,
    )

    # Get real inputs
    images = torch.load("/home/cglagovich/llama-models/ocr_vision_input_images.pt")
    ars = torch.load("/home/cglagovich/llama-models/ocr_vision_input_aspect_ratios.pt")

    reference_output = reference_model(images, ars)
    tt_out = tt_model(images, ars)
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tt_output_torch = tt_output_torch[0, :, :, :].view(reference_output.shape)

    logger.info(f"Reference output shape: {reference_output.shape}")
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Llama_Attention Passed!")
    else:
        logger.warning(f"Llama_Attention Failed!")
        all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
