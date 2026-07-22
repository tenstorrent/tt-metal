# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.minimal_matmul_descriptor import bge_qkv_minimal_matmul_descriptor


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000}], indirect=True)
def test_qkv_descriptor_parity(device):
    torch.manual_seed(0)
    x_pt = torch.randn((6, 1, 8192, 1024), dtype=torch.bfloat16)
    w_pt = torch.randn((1024, 3072), dtype=torch.bfloat16)
    b_pt = torch.randn((1, 3072), dtype=torch.bfloat16)

    x = ttnn.from_torch(x_pt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(w_pt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=16,
        K_block_size=8,
        N_block_size=4,
        subblock_h=4,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    expected = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        fused_activation=None,
        config=cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ck,
    )
    actual = bge_qkv_minimal_matmul_descriptor(
        x,
        w,
        bias_tensor=b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.synchronize_device(device)

    expected_pt = ttnn.to_torch(expected)
    actual_pt = ttnn.to_torch(actual)
    passed, pcc = comp_pcc(expected_pt, actual_pt, 0.999)
    logger.info(f"QKV_DESCRIPTOR_PCC={pcc}")
    assert passed, f"QKV descriptor parity failed: PCC={pcc}"

    # A stable cache count on the third descriptor launch proves warm reuse.
    cache_before = device.num_program_cache_entries()
    _ = bge_qkv_minimal_matmul_descriptor(
        x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.synchronize_device(device)
    cache_after_second = device.num_program_cache_entries()
    _ = bge_qkv_minimal_matmul_descriptor(
        x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.synchronize_device(device)
    cache_after_third = device.num_program_cache_entries()
    logger.info("QKV_DESCRIPTOR_CACHE entries: " f"{cache_before} -> {cache_after_second} -> {cache_after_third}")
    assert (
        cache_after_third == cache_after_second
    ), f"descriptor program cache grew on warm launch: {cache_after_second} -> {cache_after_third}"

    def trace_wall(fn, iterations=20):
        captured_output = fn()
        ttnn.synchronize_device(device)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        captured_output = fn()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - start) * 1e3)
        replayed_pt = ttnn.to_torch(captured_output)
        ttnn.release_trace(device, trace_id)
        samples.sort()
        return replayed_pt, samples[0], samples[len(samples) // 2]

    replayed_pt, descriptor_min, descriptor_median = trace_wall(
        lambda: bge_qkv_minimal_matmul_descriptor(
            x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )
    )
    replay_passed, replay_pcc = comp_pcc(expected_pt, replayed_pt, 0.999)
    logger.info(f"QKV_DESCRIPTOR_TRACE_PCC={replay_pcc}")
    assert replay_passed, f"descriptor trace replay parity failed: PCC={replay_pcc}"

    _, stock_min, stock_median = trace_wall(
        lambda: ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=b,
            fused_activation=None,
            config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
        )
    )
    logger.info(
        "QKV_TRACED_HOST_WALL_MS "
        f"descriptor min={descriptor_min:.3f} median={descriptor_median:.3f}; "
        f"stock min={stock_min:.3f} median={stock_median:.3f}; "
        f"median_delta={descriptor_median - stock_median:+.3f}"
    )
