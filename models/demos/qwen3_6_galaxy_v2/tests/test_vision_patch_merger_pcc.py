# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (8/N) PCC: PatchMerger on BH GLX 8x4.

Tests the final block in the vision encoder pipeline: LayerNorm + reshape +
Linear(4608) + GELU + Linear(5120). Weights replicated across the mesh.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_6_galaxy_v2.tt.vision_patch_merger import Qwen36VisionPatchMergerTP


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
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("seq_len", (196,))  # 14*14 input, post-merger 49 tokens
def test_vision_patch_merger_qwen36(seq_len, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=seq_len,
    )
    vc = model_args.hf_config.vision_config
    H = vc.hidden_size
    OUT = vc.out_hidden_size  # 5120
    SMS = vc.spatial_merge_size  # 2
    logger.info(f"qwen3.6 PatchMerger on {mesh_device.shape}: H={H}, OUT={OUT}, SMS={SMS}, seq_len={seq_len}")

    reference_full = model_args.reference_vision_model()
    reference_merger = reference_full.merger
    hf_state = reference_merger.state_dict()
    logger.info(f"reference merger state-dict keys: {sorted(hf_state.keys())}")

    tt_merger = Qwen36VisionPatchMergerTP(
        mesh_device=mesh_device,
        state_dict=hf_state,
        hidden_size=H,
        spatial_merge_size=SMS,
        out_hidden_size=OUT,
        norm_eps=1e-6,
        dtype=ttnn.bfloat16,
    )

    torch_input = torch.randn(seq_len, H, dtype=torch.float32)
    reference_output = reference_merger(torch_input)  # [seq // 4, 5120]

    tt_input = ttnn.from_torch(
        torch_input.view(1, 1, seq_len, H),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"Running TT PatchMerger on {mesh_device.shape}")
    tt_output = tt_merger.forward(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"tt_output_torch shape: {tuple(tt_output_torch.shape)}")
    # shape: [num_devices, 1, seq_len // unit, OUT]; take chip 0
    if tt_output_torch.dim() == 4:
        tt_output_torch = tt_output_torch[0, 0, : seq_len // (SMS * SMS), :OUT]
    else:
        tt_output_torch = tt_output_torch[0, : seq_len // (SMS * SMS), :OUT]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 vision PatchMerger PCC {pcc_required} not met: {pcc_message}"
