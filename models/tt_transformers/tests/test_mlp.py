# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (64 * 1024, 32 * 1024, 512, 32),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_mlp_inference(seq_len, batch_size, mesh_device, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    tt_model = MLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(
            (
                tt_model.model_config["MLP_ACT_MEMCFG"]
                if model_args.is_galaxy
                else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG
        ),
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run MLP")
    tt_output = tt_model(tt_input, mode)

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
        logger.info("MLP Passed!")
    else:
        logger.warning("MLP Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
