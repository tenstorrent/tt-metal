# SPDX-License-Identifier: Apache-2.0

"""Gated parity probe for the model-local BGE encoder SDPA descriptor.

This file is intentionally not part of normal coverage. The scaffold author did
not execute it; the next owner should first inspect descriptor binding errors,
then establish PCC/cache/trace parity before collecting performance numbers.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import (
    bge_encoder_sdpa_experimental,
    bge_encoder_sdpa_stock,
)

B, HQ, HKV, SQ, SK, DH = 6, 32, 16, 4096, 8192, 64


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 30_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_encoder_sdpa_descriptor_parity(mesh_device):
    if os.environ.get("BGE_RUN_UNVERIFIED_ENCODER_SDPA", "0") != "1":
        pytest.skip("set BGE_RUN_UNVERIFIED_ENCODER_SDPA=1 to run the unverified scaffold")

    torch.manual_seed(0)

    def make_tensor(heads: int, seq_len: int, dtype: ttnn.DataType) -> ttnn.Tensor:
        host = torch.randn(B, heads, seq_len, DH, dtype=torch.bfloat16)
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    q = make_tensor(HQ, SQ, ttnn.bfloat8_b)
    k = make_tensor(HKV, SK, ttnn.bfloat4_b)
    v = make_tensor(HKV, SK, ttnn.bfloat8_b)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    stock = bge_encoder_sdpa_stock(
        q,
        k,
        v,
        compute_kernel_config=compute_kernel_config,
    )
    experimental = bge_encoder_sdpa_experimental(q, k, v)
    ttnn.synchronize_device(mesh_device)

    stock_torch = ttnn.to_torch(stock, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B]
    experimental_torch = ttnn.to_torch(
        experimental,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:B]
    matches, message = comp_pcc(stock_torch, experimental_torch, 0.999)
    assert matches, message
    print(f"\nPARITY_PCC first-launch: {message}")

    # Milestone 2: second-launch program-cache reuse (warm launch, no recompile).
    import time

    from loguru import logger

    n0 = mesh_device.num_program_cache_entries()
    _ = bge_encoder_sdpa_experimental(q, k, v)
    ttnn.synchronize_device(mesh_device)
    n1 = mesh_device.num_program_cache_entries()
    logger.info(f"PROGRAM_CACHE entries after 1st extra warm launch: {n0} -> {n1} (want stable/no-growth on 3rd)")
    _ = bge_encoder_sdpa_experimental(q, k, v)
    ttnn.synchronize_device(mesh_device)
    n2 = mesh_device.num_program_cache_entries()
    logger.info(f"PROGRAM_CACHE entries after 2nd extra warm launch: {n2} (delta {n2 - n1})")
    assert n2 == n1, f"program cache still growing on warm launch: {n1} -> {n2}"

    # Milestone 3: trace capture/replay + device-duration parity vs stock.
    def trace_wall(fn):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(mesh_device, tid)
        ts.sort()
        return ts[0], ts[len(ts) // 2]

    exp_min, exp_med = trace_wall(lambda: bge_encoder_sdpa_experimental(q, k, v))
    stk_min, stk_med = trace_wall(lambda: bge_encoder_sdpa_stock(q, k, v, compute_kernel_config=compute_kernel_config))
    # NOTE: these are TRACED HOST WALL (perf_counter around blocking execute_trace),
    # NOT signpost-filtered device-kernel duration. This is traced-wall parity only;
    # do not compare directly to the ~28.9ms SDPA device duration. A device profile
    # (TT_METAL_DEVICE_PROFILER=1 + signposts) is required for device-time claims.
    logger.info(f"TRACED HOST WALL experimental: min={exp_min:.3f} med={exp_med:.3f} ms")
    logger.info(f"TRACED HOST WALL stock:        min={stk_min:.3f} med={stk_med:.3f} ms")
    logger.info(f"DELTA exp-stock (host wall): min={exp_min - stk_min:+.3f} ms ({100*(exp_min-stk_min)/stk_min:+.1f}%)")
