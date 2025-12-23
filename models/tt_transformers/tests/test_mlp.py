# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import Prefetcher


@torch.no_grad()
@pytest.mark.parametrize(
    "use_prefetcher",
    (True, False),
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
@pytest.mark.parametrize(
    "seq_len",
    (64 * 1024, 32 * 1024, 512, 32),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc, use_prefetcher):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Setup prefetcher
    num_tensors = 3 if mode == "decode" else 0
    prefetcher = Prefetcher(mesh_device, num_tensors=num_tensors, num_layers=1) if use_prefetcher else None

    if use_prefetcher:
        prefetcher.init(mode)

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=128,
        cache_hf=True,
        prefetcher=prefetcher if use_prefetcher else None,
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)
    if model_args.is_90b:
        # float32 ~3x faster than bfloat16.
        # bfloat16 fails on CI (32k and 64k seq_len) with "This test seems to have hung... Timing out test case"
        reference_model.to(torch.float32)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
        prefetcher=prefetcher,
    )

    # Run prefetcher if it is used
    if prefetcher is not None and mode == "decode":
        prefetcher.prefetch()
        prefetcher.run()

    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )
    reference_output = reference_model(torch_input)

    def get_input_memory_config():
        if mode != "decode":
            return ttnn.DRAM_MEMORY_CONFIG

        if model_args.is_galaxy:
            return tt_model.model_config["MLP_ACT_MEMCFG"]

        if prefetcher is not None:
            return model_args.model_config["PREFETCHER_SHARDED_MLP_INPUT_RING_MEMCFG"]

        return model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=get_input_memory_config(),
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
