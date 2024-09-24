# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.llama31_8b_N300.tt.llama_mlp import TtLlamaMLP
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        4096,
        # 1024,
        # 512,
        # 128,
        # 32,
    ),
)
def test_llama_mlp_inference(mesh_device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(device=mesh_device.get_devices()[0])
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaMLP(
        device_mesh=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(1, 1, seq_len, 4096)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Compilation pass for Llama_MLP")
    mode = "decode" if seq_len <= 32 else "prefill"
    tt_output = tt_model(tt_input, mode)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info("Performance pass for Llama_MLP")
    tt_output = tt_model(tt_input, mode)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")

    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
