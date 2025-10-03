# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.grok.reference.llama_clone import FeedForward
from models.demos.grok.tt.mlp import MLP
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

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1

    logger.info(f"Loading model weights")
    state_dict = model_args.load_weights_to_state_dict_no_experts()
    logger.info(f"Model weights loaded")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    logger.info(f"Loading reference model weights")
    # Create reference model using llama_clone.py FeedForward
    reference_model = FeedForward()
    reference_model.load_state_dict(partial_state_dict)
    logger.info(f"Reference model weights loaded")

    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path="./model_cache/",  # Use dummy weights
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    torch_input = torch.randn(1, 1, 32, model_args.dim, dtype=torch.float32)
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.num_devices == 32 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(model_args.model_config["MLP_ACT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG),
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run Grok MLP")
    tt_output = tt_model(tt_input)

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
