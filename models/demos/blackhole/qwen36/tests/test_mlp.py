# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.blackhole.qwen36.tt.vision.vision_mlp import MLP
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "rows",
    (
        14336,
        # 14308, # from 3B test image
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mlp_inference(rows, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = "prefill"  # Vision processing is prefill only (generating token embeddings)

    model_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=rows)
    reference_model = model_args.reference_mlp()
    state_dict = convert_hf_to_meta(reference_model.state_dict(), model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("MLP", 0)
    state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=None,  # Don't cache random weights
        layer_num=0,
    )
    torch_input = torch.randn(1, 1, rows, model_args.hf_config.vision_config.hidden_size, dtype=torch.bfloat16)
    reference_output = reference_model(torch_input)
    # The TP MLP consumes a replicated input (the wrapping DistributedLayerNorm
    # would have produced this in a full block).
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run MLP")
    tt_output = tt_model(tt_input, mode)

    # The TP MLP output is fractured along dim=3 (the hidden dim); concat along
    # that axis to reassemble the full output.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
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
