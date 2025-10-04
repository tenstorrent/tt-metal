# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.grok.tt.expert_mlp import ExpertMLP
from models.demos.grok.tt.model_config import TtModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_grok_mlp_inference(mesh_device):
    dtype = ttnn.bfloat8_b
    mode = "decode"

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    # first_layer_prefix = model_args.get_state_dict_prefix("ExpertMLP", 0)
    # partial_state_dict = {
    #     k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    # }

    # logger.info(f"Loading reference model weights")
    # Create reference model using llama_clone.py FeedForward
    # reference_model = MoE()
    # reference_model.load_state_dict(partial_state_dict)
    # logger.info(f"Reference model weights loaded")

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1

    torch_input = torch.load("torch_input.pt")
    reference_output = torch.load("reference_output.pt")

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3),
            mesh_shape=(8, 4),
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        # memory_config=(model_args.model_config["MLP_ACT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input = ttnn.repeat(tt_input, repeat_dims=(1, 8, 1, 1))

    logger.info(f"Loading model weights")
    state_dict = model_args.load_experts_weights_to_state_dict(dict())
    logger.info(f"Model weights loaded")

    tt_ccl = TT_CCL(mesh_device)
    tt_model = ExpertMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtypes={
            "w1": ttnn.bfloat8_b,
            "w2": ttnn.bfloat8_b,
            "w3": ttnn.bfloat8_b,
        },
    )

    # torch_input = torch.randn(1, 1, 32, model_args.dim, dtype=torch.float32)
    # torch.save(torch_input, "torch_input.pt")
    # logger.info(f"Reference model input shape: {torch_input.shape}")
    # reference_output = reference_model(torch_input)
    # torch.save(reference_output, "reference_output.pt")
    # logger.info(f"Reference model output shape: {reference_output.shape}")

    logger.info("Run Grok MLP")
    tt_output = tt_model(tt_input)
    breakpoint()

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    tt_output_torch = tt_output_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Grok MLP Passed!")
    else:
        logger.warning("Grok MLP Failed!")

    assert passing, f"Grok MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
