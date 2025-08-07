# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.demos.t3000.mixtral8x7b.reference.model import RMSNorm as RefRMSNorm
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh

# pytest models/tt_transformers/tests/mixtral/test_mixtral_rms_norm.py


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "t3k_mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_rms_norm_inference(
    max_seq_len,
    batch_size,
    mode,
    t3k_mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(t3k_mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    first_layer_prefix = state_dict_prefix + "ffn_norm."
    partial_state_dict = {k[-6:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    reference_model = RefRMSNorm(dim=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    # Create the inner RMSNormxw
    tt_inner_norm = RMSNorm(
        device=t3k_mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="ffn_norm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    input = torch.rand(1, 1, 32, 4096)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input,
        device=t3k_mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    tt_output = tt_inner_norm(tt_input, mode="prefill")
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mixtral_rms_norm Passed!")
    else:
        logger.warning("Mixtral_rms_norm Failed!")

    assert passing, f"Mixtral_rms_norm output does not meet PCC requirement {0.99}."
