# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.lm_head import LMHead
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import ColumnParallelLinear
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (32,),
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
def test_llama_lm_head_inference(mesh_device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    mesh_device.enable_async(False)

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        "weight": state_dict["output.weight"],
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = ColumnParallelLinear(model_args.dim, model_args.vocab_size, bias=False, init_method=lambda x: x)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix="",
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        memory_config=model_args.model_config["LM_HEAD_INPUT_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run Llama_LM_Head")
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    print(f"{tt_output_torch.shape=}")
    print(f"{reference_output.shape=}")

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama_LM_Head Passed!")
    else:
        logger.warning("Llama_LM_Head Failed!")

    assert passing, f"Llama_LM_Head output does not meet PCC requirement {pcc_required}: {pcc_message}."
