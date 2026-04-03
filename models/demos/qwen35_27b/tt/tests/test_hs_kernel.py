# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validate GDN fused kernel with HEIGHT_SHARDED L1 state.
Compares output against DRAM state baseline.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


def pcc(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    return ((a_m * b_m).sum() / (a_m.norm() * b_m.norm()).clamp(min=1e-8)).item()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_gdn_kernel_height_sharded(mesh_device):
    """Run GDN fused kernel with DRAM state, then HEIGHT_SHARDED L1 state, compare."""
    B = 32
    Nv_TP = 12
    Nk_TP = 4
    Dk = 128
    Dv = 128
    num_pairs = B * Nv_TP  # 384
    repeat_factor = Nv_TP // Nk_TP  # 3
    key_dim_tp = Dk * Nk_TP  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # 2560
    NUM_CORES = 96  # Must divide num_pairs (384/96=4) and total_rows (49152/96=512)

    def to_mesh(t, mem=None):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
        )

    torch.manual_seed(42)

    # Create test tensors
    conv_out = to_mesh(torch.randn(1, B, qkv_dim_tp, dtype=torch.bfloat16))
    a_tt = to_mesh(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    b_tt = to_mesh(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    neg_exp_A = to_mesh(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    dt_bias = to_mesh(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    norm_w = to_mesh(torch.randn(1, 1, Dv, dtype=torch.bfloat16))

    scale_val = 1.0 / math.sqrt(Dk)
    scale_tt = to_mesh(torch.full((1, 1, 1), scale_val, dtype=torch.bfloat16))
    rms_scale_tt = to_mesh(torch.full((1, 1, 1), math.sqrt(Dv), dtype=torch.bfloat16))
    rms_eps_tt = to_mesh(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.bfloat16))

    # Initial state (same for both runs)
    state_init = torch.randn(num_pairs, Dk, Dv, dtype=torch.bfloat16)

    # ---- Run 1: DRAM state (baseline) ----
    logger.info("Running GDN kernel with DRAM state...")
    state_dram = to_mesh(state_init.clone())
    output_dram = to_mesh(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))

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
        state_dram,
        output_dram,
        num_pairs=num_pairs,
        num_cores=min(96, num_pairs),
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    ttnn.synchronize_device(mesh_device)

    out_dram_cpu = ttnn.to_torch(output_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    state_dram_cpu = ttnn.to_torch(state_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"DRAM output norm: {out_dram_cpu.norm():.4f}")

    # ---- Run 2: HEIGHT_SHARDED L1 state ----
    logger.info("Running GDN kernel with HEIGHT_SHARDED L1 state...")

    # Create HEIGHT_SHARDED config on 96 cores
    total_rows = num_pairs * Dk  # 384 * 128 = 49152
    shard_h = total_rows // NUM_CORES  # 512

    # Build core grid for 96 cores: (0,0)-(7,7) = 64 cores + (0,8)-(7,10) = 32+...
    # Actually 96 cores in row-major: 96 = 8*12 but grid is 11x10=110
    # 96 cores: rows 0-7 full (8*10=80) + row 8 cols 0-7 (16) = too many
    # Let's do 96 = 8*10 + 16 = (0,0)-(7,9) [80] + (8,0)-(9,7) [16] = 96
    cg = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9)),  # 80 cores
            ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(9, 7)),  # 16 cores
        ]
    )
    assert cg.num_cores() == NUM_CORES, f"Expected {NUM_CORES} cores, got {cg.num_cores()}"

    hs_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cg, [shard_h, Dv], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Convert state to HEIGHT_SHARDED
    state_dram2 = to_mesh(state_init.clone())
    state_hs = ttnn.to_memory_config(state_dram2, hs_cfg)
    ttnn.deallocate(state_dram2)
    output_hs = to_mesh(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))

    logger.info(f"State HS: mem={state_hs.memory_config().memory_layout}, buf={state_hs.memory_config().buffer_type}")

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
        state_hs,
        output_hs,
        num_pairs=num_pairs,
        num_cores=NUM_CORES,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    ttnn.synchronize_device(mesh_device)

    out_hs_cpu = ttnn.to_torch(output_hs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    state_hs_cpu = ttnn.to_torch(state_hs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"HS output norm: {out_hs_cpu.norm():.4f}")

    # ---- Compare ----
    out_pcc = pcc(out_dram_cpu[0:num_pairs], out_hs_cpu[0:num_pairs])
    state_pcc = pcc(state_dram_cpu[0:num_pairs], state_hs_cpu[0:num_pairs])

    logger.info(f"Output PCC:  {out_pcc:.6f}")
    logger.info(f"State PCC:   {state_pcc:.6f}")

    assert out_pcc > 0.999, f"Output PCC too low: {out_pcc}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc}"
    logger.info("PASSED: HEIGHT_SHARDED kernel matches DRAM baseline")
