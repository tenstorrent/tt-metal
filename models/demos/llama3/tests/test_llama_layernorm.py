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
from models.demos.wormhole.llama31_8b_N300.tt.llama_layernorm import TtLayerNorm  # Updated import for LayerNorm
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        # 32 * 1024,
        # 32,
        2048,
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_layernorm_inference(mesh_device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    width = 1280  # Hard coded in model. TODO: Bring this into model_config
    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.transformer.resblocks.0.ln_1."
    # TODO: regex match for this / filter dict keys
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype
    # Initialize PyTorch's LayerNorm as the reference model
    # TODO: What are the shapes? Where do we get them?
    reference_model = llama_reference_mod.LayerNorm(
        normalized_shape=width,
        eps=model_args.norm_eps,
    )
    reference_model.load_state_dict(partial_state_dict)

    # Initialize the custom LayerNorm model
    tt_model = TtLayerNorm(
        device=mesh_device,
        dim=width,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        weight_dtype=dtype,
        eps=model_args.norm_eps,
    )

    # Generate random input
    # torch_input = torch.randn(1, seq_len, width)  # Adjusted dimensions for LayerNorm
    # torch_input = torch.load("layer_30_intermediate.pt")
    # torch_input = torch.load("/home/cglagovich/tt-metal/layer_31_intermediate.pt")
    torch_input = torch.load(
        "/home/cglagovich/tt-metal/models/demos/t3000/llama2_70b/reference/llama-models/image_transformer_32L_x.pt"
    )

    # Reference output using PyTorch's LayerNorm
    reference_output = reference_model(torch_input)

    # Convert input to ttnn tensor
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Compilation pass for LayerNorm")
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )  # Adjusted dim for LayerNorm
    tt_outputs = torch.chunk(tt_output_torch, model_args.num_devices, dim=-1)

    # Compare outputs
    pcc_required = 0.99
    for idx, tt_output in enumerate(tt_outputs):
        passing, pcc_message = comp_pcc(reference_output, tt_output, pcc_required)

        logger.info(comp_allclose(reference_output, tt_output))
        logger.info(pcc_message)

        if passing:
            logger.info("LayerNorm on device {idx} Passed!")
        else:
            logger.warning("LayerNorm {idx} Failed!")

        assert passing, f"LayerNorm output does not meet PCC requirement {pcc_required}: {pcc_message}."
