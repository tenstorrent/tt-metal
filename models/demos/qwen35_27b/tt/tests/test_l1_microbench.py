# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Micro-benchmark: GDN kernel with DRAM state vs L1 state.
Measures raw kernel speedup from keeping rec_states in L1.
"""

import math
import os
import time

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
def test_gdn_kernel_dram_vs_l1(mesh_device):
    """Compare GDN fused kernel time with DRAM vs L1 INTERLEAVED state."""
    B = 32
    Nv_TP = 12
    Nk_TP = 4
    Dk = 128
    Dv = 128
    num_pairs = B * Nv_TP  # 384
    repeat_factor = Nv_TP // Nk_TP  # 3
    key_dim_tp = Dk * Nk_TP  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # 2560

    def to_mesh(t, mem=None):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create test tensors matching real GDN shapes
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
    fused_output = to_mesh(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))

    # DRAM state
    state_dram = to_mesh(torch.randn(num_pairs, Dk, Dv, dtype=torch.bfloat16))
    # L1 INTERLEAVED state
    state_l1 = to_mesh(torch.randn(num_pairs, Dk, Dv, dtype=torch.bfloat16), mem=ttnn.L1_MEMORY_CONFIG)

    logger.info(f"State shape: [{num_pairs}, {Dk}, {Dv}] = {num_pairs*Dk*Dv*2/1e6:.1f} MB")
    logger.info(f"DRAM state: {state_dram.memory_config()}")
    logger.info(f"L1 state: {state_l1.memory_config()}")

    N = 20

    def run_kernel(state, label):
        # Warmup
        for _ in range(3):
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
                fused_output,
                num_pairs=num_pairs,
                num_cores=min(96, num_pairs),
                Nv_TP=Nv_TP,
                Nk_TP=Nk_TP,
                repeat_factor=repeat_factor,
                key_dim_tp=key_dim_tp,
            )
        ttnn.synchronize_device(mesh_device)

        t0 = time.time()
        for _ in range(N):
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
                fused_output,
                num_pairs=num_pairs,
                num_cores=min(96, num_pairs),
                Nv_TP=Nv_TP,
                Nk_TP=Nk_TP,
                repeat_factor=repeat_factor,
                key_dim_tp=key_dim_tp,
            )
        ttnn.synchronize_device(mesh_device)
        ms = (time.time() - t0) / N * 1000
        logger.info(f"  {label}: {ms:.3f} ms/call")
        return ms

    logger.info("\n=== GDN Kernel Benchmark ===")
    dram_ms = run_kernel(state_dram, "DRAM state")
    l1_ms = run_kernel(state_l1, "L1 state  ")

    # Copy overhead
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(N):
        ttnn.copy(state_dram, state_l1)
        ttnn.copy(state_l1, state_dram)
    ttnn.synchronize_device(mesh_device)
    copy_ms = (time.time() - t0) / N * 1000

    delta = dram_ms - l1_ms
    logger.info(f"\n{'='*50}")
    logger.info(f"DRAM state kernel:  {dram_ms:.3f} ms")
    logger.info(f"L1 state kernel:    {l1_ms:.3f} ms")
    logger.info(f"Kernel speedup:     {delta:.3f} ms ({delta/dram_ms*100:.1f}%)")
    logger.info(f"Copy round-trip:    {copy_ms:.3f} ms")
    logger.info(f"48 layers savings:  {delta*48:.1f} ms")
    logger.info(f"Swap cost (8x6):    {copy_ms/2*8:.1f} ms")
    logger.info(f"Net benefit:        {delta*48 - copy_ms/2*8:.1f} ms")
    logger.info(f"{'='*50}")
