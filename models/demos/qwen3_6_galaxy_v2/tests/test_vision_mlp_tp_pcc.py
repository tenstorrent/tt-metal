# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (3/N): vision MLP with REAL TP=8 on row axis (tt_dit primitives).

This replaces the "replicate-32" baseline test with a proper TP=8 layout.
Weights sharded along the K dim across cluster_axis=0 (8 rows).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_mlp_tp import Qwen36VisionMlpTP
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("rows", (1024,))  # smaller seq for first iteration
@pytest.mark.parametrize("layer_num", (0,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_mlp_tp_qwen36(rows, layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=rows,
    )
    vc = model_args.hf_config.vision_config
    H, I = vc.hidden_size, vc.intermediate_size
    logger.info(f"qwen3.6 vision MLP TP=8 on {mesh_device.shape}: hidden={H}, intermediate={I}")

    # Pull real qwen3.6 weights for one MLP layer
    reference_full = model_args.reference_vision_model()
    reference_mlp = reference_full.blocks[layer_num].mlp

    # tt_dit Module weight-load expects {attr_name: tensor}.
    # HF keys are linear_fc1.{weight,bias} / linear_fc2.{weight,bias}.
    hf_state = reference_mlp.state_dict()

    # Build CCLManager for the reduce_scatter / all_gather inside the MLP.
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_model = Qwen36VisionMlpTP(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        state_dict=hf_state,
        hidden_size=H,
        intermediate_size=I,
        tp_mesh_axis=0,
        dtype=ttnn.bfloat16,
    )

    torch_input = torch.randn(1, 1, rows, H)
    reference_output = reference_mlp(torch_input)

    # Input replicated across the full mesh (TP-axis-0 + col-axis-1).
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Running TT MLP TP=8 on {mesh_device.shape}")
    tt_output = tt_model(tt_input)

    # Output is replicated across the whole mesh. Pull chip 0's copy.
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tt_output_torch = tt_output_torch[:1, :, :, :H]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 vision MLP TP=8 PCC {pcc_required} not met: {pcc_message}"
