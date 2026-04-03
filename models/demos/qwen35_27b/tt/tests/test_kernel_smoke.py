# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Minimal kernel smoke test using pytest fixtures (same device setup as e2e).
"""
import os

os.environ["TT_SKIP_CB_CLASH_CHECK"] = "1"

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_kernel_smoke(mesh_device):
    """Minimal GDN kernel call — same params as e2e decode."""
    B = 32
    Nv_TP, Nk_TP, Dk, Dv = 12, 4, 128, 128
    num_pairs = B * Nv_TP  # 384 — same as e2e
    repeat_factor = Nv_TP // Nk_TP  # 3
    key_dim_tp = Dk * Nk_TP  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # 2560

    def to_mesh(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    torch.manual_seed(42)
    conv_out = to_mesh(torch.randn(1, B, qkv_dim_tp, dtype=torch.bfloat16))
    a_tt = to_mesh(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    b_tt = to_mesh(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    neg_exp_A = to_mesh(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    dt_bias = to_mesh(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    norm_w = to_mesh(torch.randn(1, 1, Dv, dtype=torch.bfloat16))
    scale_tt = to_mesh(torch.full((1, 1, 1), 1.0 / math.sqrt(Dk), dtype=torch.bfloat16))
    rms_scale_tt = to_mesh(torch.full((1, 1, 1), math.sqrt(Dv), dtype=torch.bfloat16))
    rms_eps_tt = to_mesh(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.bfloat16))
    state = to_mesh(torch.randn(num_pairs, Dk, Dv, dtype=torch.bfloat16))
    output = to_mesh(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))

    nc = min(96, num_pairs)  # same as e2e: min(96, 384) = 96
    logger.info(f"Calling kernel: {num_pairs} pairs, {nc} cores")
    gdn_full_fused_inplace(
        conv_out,
        a_tt,
        b_tt,
        neg_exp_A,
        dt_bias,
        norm_w,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state,
        output,
        num_pairs=num_pairs,
        num_cores=nc,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    ttnn.synchronize_device(mesh_device)

    out_cpu = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"Output norm: {out_cpu[:num_pairs].norm():.4f}")
    assert out_cpu[:num_pairs].abs().max() > 1e-6, "Output is all zeros!"
    logger.info("SMOKE TEST PASSED")
