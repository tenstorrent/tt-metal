# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule PCC test for MoonVisionLayerNorm.

Compares ttnn.layer_norm against the HF reference torch.nn.LayerNorm
pulled out of K2.6's MoonViTEncoderLayer.norm0. The MoonViT encoder
uses LayerNorm with eps=1e-5 at four sites in the vision tower — they
all share the same shape, so a single PCC test at this hidden dim
covers all of them.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_layernorm.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.layernorm import MoonVisionLayerNorm


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [256, 1024])
def test_moonvit_final_layernorm(mesh_device, model_args, seq_len):
    """encoder.final_layernorm (used post-block-stack, pre-patch_merger).

    Mechanically identical to the per-block LayerNorm — same hidden dim,
    same eps=1e-5 — but pinning a separate test so a future bug in the
    extraction (wrong attribute path) is caught at this position.
    """
    pcc_threshold = 0.99

    ref_ln = model_args.reference_final_layernorm()
    ref_ln_fp32 = ref_ln.float()
    assert isinstance(ref_ln, torch.nn.LayerNorm)
    assert ref_ln.normalized_shape[0] == model_args.hidden_size

    tt_ln = MoonVisionLayerNorm.from_torch(mesh_device=mesh_device, ref=ref_ln, dtype=ttnn.bfloat16)

    torch.manual_seed(0)
    x_pt_fp32 = torch.randn(1, 1, seq_len, model_args.hidden_size, dtype=torch.float32)
    x_pt_bf16 = x_pt_fp32.to(torch.bfloat16)

    ref_out = ref_ln_fp32(x_pt_fp32)

    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_pt_bf16,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_out = ttnn.to_torch(
        tt_ln(x_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and tt_out.shape[0] != ref_out.shape[0]:
        tt_out = tt_out[: ref_out.shape[0]]

    passing, pcc_msg = comp_pcc(ref_out, tt_out, pcc_threshold)
    logger.info(f"[final_ln seq={seq_len}] {comp_allclose(ref_out, tt_out)} {pcc_msg}")
    assert passing, f"final_layernorm PCC below threshold at seq_len={seq_len}: {pcc_msg}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("which", ["norm0", "norm1"])
def test_moonvit_layernorm(mesh_device, model_args, seq_len, which):
    """LayerNorm forward matches HF reference at PCC >= 0.99."""

    pcc_threshold = 0.99

    # 1. HF reference module (eval, cpu, bfloat16 from the checkpoint).
    ref_ln = model_args.reference_layernorm(layer_num=0, which=which)
    # Cast to fp32 for the torch reference run — gives us a cleaner golden
    # to compare bf16 ttnn output against (the small mean+norm op is
    # numerically sensitive enough that bf16 vs bf16 PCC can be lower
    # than bf16 vs fp32, counterintuitively, because fp32 ref is the
    # true mathematical result).
    ref_ln_fp32 = ref_ln.float()

    dim = ref_ln_fp32.normalized_shape[0]
    assert dim == model_args.hidden_size, f"reference dim {dim} != model_args.hidden_size {model_args.hidden_size}"

    # 2. Construct our ttnn module from the same weights.
    tt_ln = MoonVisionLayerNorm.from_torch(
        mesh_device=mesh_device,
        ref=ref_ln,
        dtype=ttnn.bfloat16,
    )

    # 3. Random input — same data, two paths.
    torch.manual_seed(0)
    x_pt_fp32 = torch.randn(1, 1, seq_len, dim, dtype=torch.float32)
    x_pt_bf16 = x_pt_fp32.to(torch.bfloat16)

    # Torch reference: fp32 input through fp32 LayerNorm.
    ref_out = ref_ln_fp32(x_pt_fp32)

    # TT-Metal: bf16 input on device.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_pt_bf16,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_out_tensor = tt_ln(x_tt)
    tt_out = ttnn.to_torch(
        tt_out_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    # If mesh-composed, the replicate path yields N copies stacked on dim 0; take the first.
    if is_mesh and tt_out.shape[0] != ref_out.shape[0]:
        tt_out = tt_out[: ref_out.shape[0]]

    # 4. PCC gate + diagnostic.
    passing, pcc_msg = comp_pcc(ref_out, tt_out, pcc_threshold)
    logger.info(f"[{which} seq={seq_len}] {comp_allclose(ref_out, tt_out)} {pcc_msg}")
    assert passing, f"PCC below threshold for which={which} seq_len={seq_len}: {pcc_msg}"
