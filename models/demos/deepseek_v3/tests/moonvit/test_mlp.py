# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule PCC test for MoonVisionMLP.

Compares fc0 -> GELU(tanh) -> fc1 against the HF MLP2 reference. The
intermediate dim 4304 isn't tile-aligned, so this test also implicitly
validates ttnn's auto-padding behavior for non-aligned linears.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_mlp.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.mlp import MoonVisionMLP


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [256, 1024])
def test_moonvit_mlp(mesh_device, model_args, seq_len):
    """MLP forward matches HF MLP2 at PCC >= 0.99."""

    pcc_threshold = 0.99

    # 1. HF reference module.
    ref_mlp = model_args.reference_mlp(layer_num=0)
    ref_mlp_fp32 = ref_mlp.float()
    assert ref_mlp.fc0.in_features == model_args.hidden_size
    assert ref_mlp.fc0.out_features == model_args.intermediate_size

    # 2. ttnn module with the same weights.
    tt_mlp = MoonVisionMLP.from_torch(mesh_device=mesh_device, ref=ref_mlp, dtype=ttnn.bfloat16)

    # 3. Input — same data, two paths.
    torch.manual_seed(0)
    x_pt_fp32 = torch.randn(1, 1, seq_len, model_args.hidden_size, dtype=torch.float32)
    x_pt_bf16 = x_pt_fp32.to(torch.bfloat16)

    ref_out = ref_mlp_fp32(x_pt_fp32)

    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_pt_bf16,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_out_tensor = tt_mlp(x_tt)
    tt_out = ttnn.to_torch(
        tt_out_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and tt_out.shape[0] != ref_out.shape[0]:
        tt_out = tt_out[: ref_out.shape[0]]

    # 4. PCC gate.
    passing, pcc_msg = comp_pcc(ref_out, tt_out, pcc_threshold)
    logger.info(f"[seq={seq_len}] {comp_allclose(ref_out, tt_out)} {pcc_msg}")
    assert passing, f"PCC below threshold for seq_len={seq_len}: {pcc_msg}"
