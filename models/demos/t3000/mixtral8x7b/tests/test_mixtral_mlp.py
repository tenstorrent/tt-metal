# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
from loguru import logger

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_mlp import TtMixtralMLP
from models.demos.t3000.mixtral8x7b.reference.model import FeedForward, RMSNorm
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    skip_for_wormhole_b0,
)


def test_mixtral_mlp_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    # Specify different dtypes for each feedForward weights
    dtypes = {
        "w1": ttnn.bfloat4_b,
        "w2": ttnn.bfloat8_b,
        "w3": ttnn.bfloat4_b,
    }

    model_args = TtModelArgs(t3k_mesh_device.get_device(0))
    state_dict = model_args.load_state_dict()

    tt_model = TtMixtralMLP(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtypes=dtypes,
    )

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }
    partial_state_dict_ref = {k[32:]: v for k, v in partial_state_dict.items() if f"experts.{0}" in k}
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(partial_state_dict_ref)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    rms_state_dict = {k[18:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    rms = RMSNorm(dim=model_args.dim)
    rms.load_state_dict(rms_state_dict)

    torch_input = (torch.rand(1, 1, 32, model_args.dim) * 2) - 1
    torch_input = rms(torch_input)  # apply rmsnorm to input

    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=t3k_mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)
    if passing:
        logger.info("Mixtral_MLP Passed!")
    else:
        logger.warning("Mixtral_MLP Failed!")

    assert passing, f"Mixtral_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
