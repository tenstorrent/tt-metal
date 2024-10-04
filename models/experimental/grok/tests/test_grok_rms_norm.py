# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.experimental.grok.tt.grok_rms_norm import TtRMSNorm, TtRMSNormSharded
from models.experimental.grok.reference.model import RMSNorm
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_grok_rms_norm_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    t3k_mesh_device.enable_async(True)

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=os.getenv("CI") == "true")
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.pre_attn_norm."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if (k.startswith(key_start))}

    reference_model = RMSNorm(hidden_size=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNorm(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        dtype=dtype,
        layer_num=0,
        weight_key="pre_attn_norm",
    )
    input = torch.rand(1, 1, 32, model_args.dim)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input,
        device=t3k_mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Grok_rms_norm Passed!")
    else:
        logger.warning("Grok_rms_norm Failed!")

    assert passing, f"Grok_rms_norm output does not meet PCC requirement {0.99}."


def test_grok_rms_norm_sharded_inference(t3k_mesh_device, use_program_cache, reset_seeds):
    t3k_mesh_device.enable_async(True)
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=os.getenv("CI") == "true")
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    key_start = "model.layers.0.pre_attn_norm."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if (k.startswith(key_start))}

    reference_model = RMSNorm(hidden_size=model_args.hidden_size)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNormSharded(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        dtype=dtype,
        layer_num=0,
        weight_key="pre_attn_norm",
    )
    input = torch.rand(1, 1, 32, model_args.dim)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input,
        device=t3k_mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Grok_rms_norm Passed!")
    else:
        logger.warning("Grok_rms_norm Failed!")

    assert passing, f"Grok_rms_norm output does not meet PCC requirement {0.99}."
