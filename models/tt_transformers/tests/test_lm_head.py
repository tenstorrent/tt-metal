# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_lm_head_inference(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        "weight": state_dict[f"{state_dict_prefix}output.weight"],
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = model_args.reference_lm_head()
    reference_model.load_state_dict(partial_state_dict)
    reference_model.to(torch.float32)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        max_columns_per_device=model_args.max_columns_per_device_lm_head,
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=model_args.model_config["LM_HEAD_INPUT_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run LM_Head")
    num_iterations = 100000
    for i in tqdm(range(num_iterations)):
        tt_output = tt_model(tt_input)

    # Concatenate the outputs
    tt_output = ttnn.concat(
        tt_output, dim=-1, memory_config=model_args.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
    )

    tt_output = tt_all_reduce(
        tt_output,
        model_args.mesh_device,
        tt_ccl,
        cluster_axis=1,
        dim=0,
        num_reduce_scatter_links=model_args.num_reduce_scatter_links,
        num_all_gather_links=model_args.num_all_gather_links,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=model_args.ccl_dtype,
        sharded=False,
        use_composite=True,
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, model_args.cluster_shape, dims=(3, 1) if model_args.is_galaxy else (1, 3)
        ),
    )
    tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.vocab_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("LM_Head Passed!")
    else:
        logger.warning("LM_Head Failed!")

    assert passing, f"LM_Head output does not meet PCC requirement {pcc_required}: {pcc_message}."
