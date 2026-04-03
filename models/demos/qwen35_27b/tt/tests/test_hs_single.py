#!/usr/bin/env python3
"""
Single-device test: GDN kernel with HEIGHT_SHARDED L1 state.
No fabric, no mesh — just one device.

Run:
    tt-smi -r 0 && cd ~/tt-metal && \
    python models/demos/qwen35_27b/tt/tests/test_hs_single.py
"""
import os

os.environ["TT_SKIP_CB_CLASH_CHECK"] = "1"

import math
import time

import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


def pcc(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    return ((a_m * b_m).sum() / (a_m.norm() * b_m.norm()).clamp(min=1e-8)).item()


def main():
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    logger.info(f"Device grid: {grid.x}×{grid.y} = {grid.x * grid.y} cores")

    B = 32
    Nv_TP = 12
    Nk_TP = 4
    Dk = 128
    Dv = 128
    num_pairs = B * Nv_TP  # 384
    repeat_factor = Nv_TP // Nk_TP  # 3
    key_dim_tp = Dk * Nk_TP  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # 2560
    NUM_CORES = 96

    def to_dev(t, mem=None):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG
        )

    torch.manual_seed(42)

    # Create inputs
    conv_out = to_dev(torch.randn(1, B, qkv_dim_tp, dtype=torch.bfloat16))
    a_tt = to_dev(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    b_tt = to_dev(torch.randn(1, B, Nv_TP, dtype=torch.bfloat16))
    neg_exp_A = to_dev(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    dt_bias = to_dev(torch.randn(1, 1, Nv_TP, dtype=torch.bfloat16))
    norm_w = to_dev(torch.randn(1, 1, Dv, dtype=torch.bfloat16))
    scale_tt = to_dev(torch.full((1, 1, 1), 1.0 / math.sqrt(Dk), dtype=torch.bfloat16))
    rms_scale_tt = to_dev(torch.full((1, 1, 1), math.sqrt(Dv), dtype=torch.bfloat16))
    rms_eps_tt = to_dev(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.bfloat16))

    state_init = torch.randn(num_pairs, Dk, Dv, dtype=torch.bfloat16)

    # ---- Test 1: DRAM state (baseline) ----
    logger.info("\n=== Test 1: DRAM state ===")
    state_dram = to_dev(state_init.clone())
    output_dram = to_dev(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))
    logger.info("Calling gdn_full_fused_inplace (DRAM)... (first call compiles kernel, may take 2-3 min)")
    import sys

    sys.stdout.flush()

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
        num_cores=NUM_CORES,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    ttnn.synchronize_device(device)

    out_dram = ttnn.to_torch(output_dram)
    state_after_dram = ttnn.to_torch(state_dram)
    logger.info(f"DRAM output norm: {out_dram[:num_pairs].norm():.4f}")
    logger.info(f"DRAM state norm:  {state_after_dram[:num_pairs].norm():.4f}")

    # ---- Test 2: HEIGHT_SHARDED L1 state ----
    logger.info("\n=== Test 2: HEIGHT_SHARDED L1 state ===")
    total_rows = num_pairs * Dk  # 49152
    shard_h = total_rows // NUM_CORES  # 512

    # Build core grid for 96 cores
    # Grid is 11×10. 96 = 11*8 + 8 = 88+8
    cr1 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7))  # 88 cores
    cr2 = ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8))  # 8 cores
    cg = ttnn.CoreRangeSet([cr1, cr2])
    logger.info(f"Core grid: {cg.num_cores()} cores, shard: [{shard_h}, {Dv}]")

    hs_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cg, [shard_h, Dv], ttnn.ShardOrientation.ROW_MAJOR),
    )

    state_dram2 = to_dev(state_init.clone())
    state_hs = ttnn.to_memory_config(state_dram2, hs_cfg)
    ttnn.deallocate(state_dram2)
    output_hs = to_dev(torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16))

    logger.info(f"HS state: mem={state_hs.memory_config().memory_layout}")
    logger.info(f"HS state: buf={state_hs.memory_config().buffer_type}")
    logger.info(f"HS state: addr=0x{state_hs.buffer_address():x}")

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
    ttnn.synchronize_device(device)

    out_hs = ttnn.to_torch(output_hs)
    state_after_hs = ttnn.to_torch(state_hs)
    logger.info(f"HS output norm:   {out_hs[:num_pairs].norm():.4f}")
    logger.info(f"HS state norm:    {state_after_hs[:num_pairs].norm():.4f}")

    # ---- Compare ----
    logger.info("\n=== Comparison ===")
    out_p = pcc(out_dram[:num_pairs], out_hs[:num_pairs])
    state_p = pcc(state_after_dram[:num_pairs], state_after_hs[:num_pairs])
    logger.info(f"Output PCC: {out_p:.6f}")
    logger.info(f"State PCC:  {state_p:.6f}")

    if out_p > 0.999 and state_p > 0.999:
        logger.info("PASSED: HEIGHT_SHARDED matches DRAM baseline!")
    else:
        logger.error(f"FAILED: PCC too low (output={out_p:.6f}, state={state_p:.6f})")

    # ---- Benchmark ----
    logger.info("\n=== Benchmark (10 iterations) ===")
    N = 10

    # Reset states
    state_dram = to_dev(state_init.clone())
    state_dram2 = to_dev(state_init.clone())
    state_hs = ttnn.to_memory_config(state_dram2, hs_cfg)
    ttnn.deallocate(state_dram2)

    # Warmup
    for _ in range(2):
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
            num_cores=NUM_CORES,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
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
    ttnn.synchronize_device(device)

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
            state_dram,
            output_dram,
            num_pairs=num_pairs,
            num_cores=NUM_CORES,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
    ttnn.synchronize_device(device)
    dram_ms = (time.time() - t0) / N * 1000

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
            state_hs,
            output_hs,
            num_pairs=num_pairs,
            num_cores=NUM_CORES,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
    ttnn.synchronize_device(device)
    hs_ms = (time.time() - t0) / N * 1000

    delta = dram_ms - hs_ms
    logger.info(f"DRAM:        {dram_ms:.3f} ms/call")
    logger.info(f"HEIGHT_SHARD: {hs_ms:.3f} ms/call")
    logger.info(f"Speedup:     {delta:.3f} ms ({delta/dram_ms*100:.1f}%)")
    logger.info(f"48 layers:   {delta*48:.1f} ms saved")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
