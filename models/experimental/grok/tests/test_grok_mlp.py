# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
from loguru import logger
import pytest

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.experimental.grok.tt.grok_mlp import TtGrokMLP
from models.experimental.grok.reference.model import MoeMLP, RMSNorm
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.timeout(500)
def test_grok_mlp_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    t3k_mesh_device.enable_async(True)
    # Specify different dtypes for each feedForward weights
    dtypes = {
        "linear": ttnn.bfloat4_b,
        "linear_1": ttnn.bfloat8_b,
        "linear_v": ttnn.bfloat4_b,
    }

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=os.getenv("CI") == "true")
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    tt_model = TtGrokMLP(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtypes=dtypes,
    )

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.moe_block.experts.0."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if k.startswith(key_start)}

    reference_model = MoeMLP(hidden_dim=model_args.hidden_size, ffn_dim=model_args.intermediate_size)
    reference_model.load_state_dict(partial_state_dict)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.pre_moe_norm."
    rms_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if k.startswith(key_start)}

    rms = RMSNorm(hidden_size=model_args.hidden_size)
    rms.load_state_dict(rms_state_dict)

    torch_input = (torch.rand(1, 1, 32, model_args.hidden_size) * 2) - 1
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

    pcc_required = 0.98  # random weights = 0.985
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)
    if passing:
        logger.info("Grok_MLP Passed!")
    else:
        logger.warning("Grok_MLP Failed!")

    assert passing, f"Grok_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
