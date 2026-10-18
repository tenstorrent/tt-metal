# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""PCC test for ``TtLfm2VlMultiModalProjector`` against ``reference/functional.lfm2_vl_projector``."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.multimodal.lfm25_vl.reference.functional import lfm2_vl_projector
from models.demos.multimodal.lfm25_vl.tt.model_config import ModelArgs
from models.demos.multimodal.lfm25_vl.tt.multi_modal_projector import TtLfm2VlMultiModalProjector


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", (1,))
def test_projector_inference(batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    prefix = "model.multi_modal_projector"
    linear_1_w = state_dict[f"{prefix}.linear_1.weight"]
    linear_2_w = state_dict[f"{prefix}.linear_2.weight"]
    linear_1_b = state_dict.get(f"{prefix}.linear_1.bias")
    linear_2_b = state_dict.get(f"{prefix}.linear_2.bias")

    num_patches = model_args.vision_num_patches
    torch_input = torch.randn(batch_size, num_patches, model_args.vision_dim, dtype=torch.float32)
    reference_output = lfm2_vl_projector(
        torch_input,
        linear_1_w,
        linear_2_w,
        linear_1_b=linear_1_b,
        linear_2_b=linear_2_b,
        downsample_factor=model_args.downsample_factor,
    )

    tt_model = TtLfm2VlMultiModalProjector(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=prefix,
        vision_dim=model_args.vision_dim,
        projector_hidden_size=model_args.projector_hidden_size,
        text_dim=model_args.dim,
        downsample_factor=model_args.downsample_factor,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        bias=model_args.projector_bias,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Run MultiModalProjector")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tt_output_torch = tt_output_torch[:batch_size]

    pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"Projector output does not meet PCC requirement {pcc_required}: {pcc_message}."
