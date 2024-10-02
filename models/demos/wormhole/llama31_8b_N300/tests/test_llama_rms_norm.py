# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (4, 2), "TG": (8, 4)}.get(os.environ.get("FAKE_DEVICE"), None)],
    indirect=True,
)
def test_llama_rms_norm_inference(mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[24:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention_norm."))}
    reference_model = RefRMSNorm(dim=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        layer_num=0,
        weight_key="attention_norm",
        weight_dtype=dtype,
    )
    input = torch.rand(1, 32, model_args.dim)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # , device, put_on_device=False)

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :1, :, :
    ].squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Llama_rms_norm Passed!")
    else:
        logger.warning("Llama_rms_norm Failed!")

    assert passing, f"Llama_rms_norm output does not meet PCC requirement {0.99}."
