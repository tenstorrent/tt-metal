# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode input RMSNorm on [1, 1, 32, 4096] (one chip): interleaved (single core) vs width-sharded multicore
(reshard in + sharded norm + reshard out) vs fp32-in / bf16-out in one op (fused typecast). Traced replay us.

    TT_VISIBLE_DEVICES=4 pytest models/demos/glm53_flash/tests/test_norm_configs.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn

N_REP = 50


def _timed(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    ttnn.release_trace(device, tid)
    return out, us


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000, "l1_small_size": 32768}], indirect=True)
def test_norm_configs(mesh_device):
    device = mesh_device
    H, B = 4096, 32
    torch.manual_seed(0)
    x = torch.randn(1, 1, B, H) * 3
    w = torch.rand(H) + 0.5
    ref = (x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)) * w
    hifi4 = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    w_t = ttnn.from_torch(w.reshape(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x16 = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x32 = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def pcc(o):
        t = ttnn.to_torch(o).float().reshape(B, H)
        return torch.corrcoef(torch.stack([ref.flatten(), t.flatten()]))[0, 1].item()

    o, us = _timed(device, lambda: ttnn.rms_norm(x16, epsilon=1e-5, weight=w_t, compute_kernel_config=hifi4))
    logger.info(f"interleaved bf16 rms_norm (1 core):               {us:7.1f} us  pcc {pcc(o):.6f}")
    o, us = _timed(
        device,
        lambda: ttnn.rms_norm(ttnn.typecast(x32, ttnn.bfloat16), epsilon=1e-5, weight=w_t, compute_kernel_config=hifi4),
    )
    logger.info(f"typecast fp32->bf16 + interleaved rms_norm:       {us:7.1f} us  pcc {pcc(o):.6f}")
    try:
        o, us = _timed(
            device,
            lambda: ttnn.rms_norm(x32, epsilon=1e-5, weight=w_t, compute_kernel_config=hifi4, dtype=ttnn.bfloat16),
        )
        logger.info(
            f"fp32-in rms_norm dtype=bf16 (one op):            {us:7.1f} us  pcc {pcc(o):.6f}  out dtype {o.dtype}"
        )
    except Exception as e:  # noqa: BLE001
        logger.info(f"fp32-in rms_norm dtype=bf16 failed: {str(e).splitlines()[0][:140]}")
    for cores_x, cores_y in ((8, 4), (8, 2), (4, 2)):
        n = cores_x * cores_y
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, cores_y - 1))})
        shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (B, H // n), ttnn.ShardOrientation.ROW_MAJOR),
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(cores_x, cores_y),
            subblock_w=max(d for d in (1, 2, 4, 8) if (H // n // 32) % d == 0),
            block_h=B // 32,
            block_w=H // n // 32,
            inplace=False,
        )
        try:

            def f():
                xs = ttnn.to_memory_config(x16, shard)
                y = ttnn.rms_norm(
                    xs, epsilon=1e-5, weight=w_t, program_config=pc, memory_config=shard, compute_kernel_config=hifi4
                )
                return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

            o, us = _timed(device, f)
            logger.info(
                f"reshard + sharded rms_norm {cores_x}x{cores_y} + reshard (3 ops):  {us:7.1f} us  pcc {pcc(o):.6f}"
            )
            o, us = _timed(
                device,
                lambda: ttnn.rms_norm(
                    ttnn.to_memory_config(x16, shard),
                    epsilon=1e-5,
                    weight=w_t,
                    program_config=pc,
                    memory_config=shard,
                    compute_kernel_config=hifi4,
                ),
            )
            logger.info(
                f"reshard + sharded rms_norm {cores_x}x{cores_y} (2 ops, sharded out):{us:7.1f} us  pcc {pcc(o):.6f}"
            )
        except Exception as e:  # noqa: BLE001
            logger.info(f"sharded {cores_x}x{cores_y} failed: {str(e).splitlines()[0][:140]}")
