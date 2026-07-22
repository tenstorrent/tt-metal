# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit
from models.demos.wormhole.bge_m3.tt.custom_ops.qkv_scatter_matmul import bge_qkv_scatter_matmul


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000}], indirect=True)
def test_qkv_scatter_descriptor_parity(device):
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn((6, 1, 8192, 1024), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    w = ttnn.from_torch(
        torch.randn((1024, 3072), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.randn((1, 3072), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

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
    fused = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        fused_activation=None,
        config=cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ck,
    )
    expected = bge_qkv_heads_headsplit(
        fused,
        num_heads=16,
        head_groups=4,
        out_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        k_out_dtype=ttnn.bfloat4_b,
        v_out_dtype=ttnn.bfloat4_b,
    )
    actual = bge_qkv_scatter_matmul(
        x,
        w,
        bias_tensor=b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.synchronize_device(device)

    expected_pt = tuple(ttnn.to_torch(t) for t in expected)
    for name, expected_tensor_pt, actual_tensor in zip(("Q", "K", "V"), expected_pt, actual):
        passed, pcc = comp_pcc(expected_tensor_pt, ttnn.to_torch(actual_tensor), 0.999)
        logger.info(f"QKV_SCATTER_{name}_PCC={pcc}")
        assert passed, f"QKV scatter {name} parity failed: PCC={pcc}"

    cache_before = device.num_program_cache_entries()
    _ = bge_qkv_scatter_matmul(x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.synchronize_device(device)
    cache_after_second = device.num_program_cache_entries()
    _ = bge_qkv_scatter_matmul(x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.synchronize_device(device)
    cache_after_third = device.num_program_cache_entries()
    logger.info(f"QKV_SCATTER_CACHE entries: {cache_before} -> {cache_after_second} -> {cache_after_third}")
    assert cache_after_third == cache_after_second

    def baseline():
        qkv = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=b,
            fused_activation=None,
            config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
        )
        outputs = bge_qkv_heads_headsplit(
            qkv,
            num_heads=16,
            head_groups=4,
            out_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            k_out_dtype=ttnn.bfloat4_b,
            v_out_dtype=ttnn.bfloat4_b,
        )
        ttnn.deallocate(qkv)
        return outputs

    def scatter():
        return bge_qkv_scatter_matmul(x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

    def trace_wall(fn, iterations=30):
        fn()
        ttnn.synchronize_device(device)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        captured = fn()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - start) * 1e3)
        replayed = tuple(ttnn.to_torch(t) for t in captured)
        ttnn.release_trace(device, trace_id)
        samples.sort()
        return replayed, samples[0], samples[len(samples) // 2]

    replayed, scatter_min, scatter_median = trace_wall(scatter)
    for name, expected_tensor_pt, replayed_tensor_pt in zip(("Q", "K", "V"), expected_pt, replayed):
        passed, pcc = comp_pcc(expected_tensor_pt, replayed_tensor_pt, 0.999)
        logger.info(f"QKV_SCATTER_TRACE_{name}_PCC={pcc}")
        assert passed, f"QKV scatter trace {name} parity failed: PCC={pcc}"
    _, baseline_min, baseline_median = trace_wall(baseline)
    logger.info(
        "QKV_SCATTER_TRACED_HOST_WALL_MS "
        f"scatter min={scatter_min:.3f} median={scatter_median:.3f}; "
        f"baseline min={baseline_min:.3f} median={baseline_median:.3f}; "
        f"median_delta={scatter_median - baseline_median:+.3f}"
    )
