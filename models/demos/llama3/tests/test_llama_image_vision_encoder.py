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
from models.demos.llama3.tt.llama_image_vision_encoder import TtLlamaVisionEncoder
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
def test_llama_vision_encoder_inference(mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    pcc = 0.99

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    return_intermediate = "3,7,15,23,30"
    return_intermediate = [int(l) for l in return_intermediate.split(",")]

    reference_model = llama_reference_mod.VisionEncoder(
        max_num_tiles=4,
        image_size=model_args.vision_chunk_size,
        patch_size=model_args.vision_patch_size,
        n_global_layers=8,
        global_model=True,
        return_intermediate=return_intermediate,
    )
    reference_model.load_state_dict(partial_state_dict, strict=True)

    all_tests_pass = True

    tt_model = TtLlamaVisionEncoder(
        mesh_device,
        state_dict,
        first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        return_intermediate=return_intermediate,
    )

    # Get real inputs
    images = torch.load("/home/cglagovich/llama-models/vision_encoder_images.pt")
    ars = torch.load("/home/cglagovich/llama-models/vision_encoder_ar.pt")

    reference_output = reference_model(images, ars)
    tt_out = tt_model(images, ars)
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tt_output_torch = tt_output_torch[0, :, :, :].view(reference_output.shape)

    INTERM_OUT = True
    if INTERM_OUT:
        # reference_output is [x] + [shuffled_int_x]
        # tt_output is [x] + [int_x]
        # To compare, we will shuffle tt_output. NOTE! This requires that the vision model shuffle its projection weights
        tt_output_shuffled = torch.zeros_like(tt_output_torch)
        tt_output_shuffled[..., : model_args.vision_dim] = tt_output_torch[..., : model_args.vision_dim]
        tt_int_x = tt_output_torch[..., model_args.vision_dim :]
        tt_int_x = (
            tt_int_x.reshape(reference_output.shape[:-1] + (5, model_args.vision_dim))
            .transpose(-1, -2)
            .reshape(reference_output.shape[:-1] + (model_args.vision_dim * 5,))
        )
        tt_output_shuffled[..., model_args.vision_dim :] = tt_int_x

        logger.info(f"Reference output shape: {reference_output.shape}")
        logger.info(f"TT output shape: {tt_output_shuffled.shape}")

        passing, pcc_message = comp_pcc(reference_output, tt_output_shuffled, pcc)

        logger.info(comp_allclose(reference_output, tt_output_shuffled))
        logger.info(f"PCC: {pcc_message}")
    else:
        logger.info(f"Reference output shape: {reference_output.shape}")
        logger.info(f"TT output shape: {tt_output_torch.shape}")

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    # reference_output = reference_model(images, ars)
    # tt_out = tt_model(images, ars)
    # tt_output_torch = [ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)) for tt in tt_out]
    # tt_output_torch = [tt[0, :, :, :].view(
    #     reference_output[0].shape
    # ) for tt in tt_output_torch] # [ batch, seq, hidden_dim]

    # for i in range(len(tt_output_torch)):
    #     logger.info(f"Reference output shape: {reference_output[i].shape}")
    #     logger.info(f"TT output shape: {tt_output_torch[i].shape}")
    #     passing, pcc_message = comp_pcc(reference_output[i], tt_output_torch[i], pcc)
    #     logger.info(comp_allclose(reference_output[i], tt_output_torch[i]))
    #     logger.info(f'PCC: {pcc_message}')
    #     # return
    # logger.info(f"Reference output shape: {reference_output.shape}")
    # logger.info(f"TT output shape: {tt_output_torch.shape}")

    # passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    # logger.info(comp_allclose(reference_output, tt_output_torch))
    # logger.info(f'PCC: {pcc_message}')

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
