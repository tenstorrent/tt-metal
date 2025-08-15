# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta


@torch.no_grad()
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
    "rows",
    (14308,),  # from 3B test image
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_patch_merger_inference(rows, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=rows)

    # Create reference model with correct dimensions
    reference_model = model_args.reference_patch_merger()

    state_dict = convert_hf_to_meta(reference_model.state_dict(), model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("PatchMerger")
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    tt_model = PatchMerger(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=None,  # Don't cache random weights
        dtype=dtype,
    )

    # Input shape should match context_dim
    torch_input = torch.randn(batch_size, 1, rows, model_args.hf_config.vision_config.hidden_size)
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run PatchMerger")
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = (
        tt_output_torch[:, 0:1, :, : model_args.hf_config.vision_config.out_hidden_size].squeeze(0).squeeze(0)
    )

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("PatchMerger Passed!")
    else:
        logger.warning("PatchMerger Failed!")

    assert passing, f"PatchMerger output does not meet PCC requirement {pcc_required}: {pcc_message}."
