"""Gemma-3-4b-it Test for Text MLP"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import os
import ttnn

from models.experimental.gemma3_4b.tt.mlp import MLP
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull

from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


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
    (2560,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference(seq_len, batch_size, reset_seeds, mesh_device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_ccl = TT_CCL(mesh_device)
    tt_model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128)

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    # # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    # first_layer_prefix = "layers.0.feed_forward"
    first_layer_prefix = tt_model_args.get_state_dict_prefix("MLP", 0)

    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = tt_model_args.reference_mlp()  # Gemma3 MLP
    reference_model.load_state_dict(partial_state_dict)

    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=tt_model_args,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=tt_model_args.get_model_config(),
        state_dict_prefix=first_layer_prefix,
    )
    torch_input = torch.randn(1, 1, seq_len)
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if tt_model_args.is_galaxy else (None, None),
            mesh_shape=tt_model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run MLP")
    tt_output = tt_model(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=tt_model_args.cluster_shape),
    )

    # tt_output_torch = tt_output_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch[0]))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("MLP Passed!")
    else:
        logger.warning("MLP Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
