# SPDX-License-Identifier: Apache-2.0
"""Encoder-SDPA chunk sweep (README specialization step 4: CB-aliasing / bigger Q).

The model-local JIT SDPA lets us size CBs independently of stock SDPA. Config-only
item-1 found Q>128 OOMs at K2048 (score CB q_chunk*k_chunk dominates). But halving
K to 1024 frees enough L1 to grow Q to 256 (est ~1057KB/core < 1440 budget). Bigger
Q = fewer Q passes (SDPA throughput is set by Sq). This sweeps (q_chunk,k_chunk)
and reports standalone traced wall + PCC vs stock. Gated by env; not normal coverage.
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import (
    bge_encoder_sdpa_stock,
    build_encoder_sdpa_descriptor,
)

B, HQ, HKV, SQ, SK, DH = 6, 32, 16, 4096, 8192, 64


@pytest.mark.parametrize("device_params", [{"trace_region_size": 40_000_000, "num_command_queues": 1}], indirect=True)
def test_encoder_sdpa_chunk_sweep(mesh_device):
    if os.environ.get("BGE_RUN_UNVERIFIED_ENCODER_SDPA", "0") != "1":
        pytest.skip("set BGE_RUN_UNVERIFIED_ENCODER_SDPA=1")
    torch.manual_seed(0)
    dev = mesh_device

    def mk(heads, seq, dt):
        return ttnn.from_torch(
            torch.randn(B, heads, seq, DH, dtype=torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    q, k, v = mk(HQ, SQ, ttnn.bfloat8_b), mk(HKV, SK, ttnn.bfloat4_b), mk(HKV, SK, ttnn.bfloat8_b)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True
    )

    stock = bge_encoder_sdpa_stock(q, k, v, compute_kernel_config=ckc)
    ttnn.synchronize_device(dev)
    stock_t = ttnn.to_torch(stock, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B]

    def trace_wall(build):
        ttnn.generic_op(build.io_tensors, build.descriptor)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.generic_op(build.io_tensors, build.descriptor)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(dev, tid)
        ts.sort()
        return ts[0], ts[len(ts) // 2]

    # (q_chunk, k_chunk). First is the parity default.
    combos_env = os.environ.get("BGE_SDPA_COMBOS", "")
    if combos_env:
        combos = [tuple(int(x) for x in c.split("x")) for c in combos_env.split(",")]
    else:
        combos = [(128, 2048), (128, 1024), (256, 1024), (192, 1024), (256, 512), (512, 512), (256, 2048)]
    logger.info(f"stock baseline established (PCC ref)")
    for qc, kc in combos:
        cfg = EncoderSDPAConfig(q_chunk_size=qc, k_chunk_size=kc)
        try:
            build = build_encoder_sdpa_descriptor(q, k, v, config=cfg)
            out = build.output
            ttnn.generic_op(build.io_tensors, build.descriptor)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B]
            _, pcc_msg = comp_pcc(stock_t, got, 0.99)
            mn, md = trace_wall(build)
            logger.info(f"q={qc:4d} k={kc:5d}: wall min={mn:7.3f} med={md:7.3f} ms  pcc={pcc_msg}")
        except Exception as e:
            logger.error(f"q={qc:4d} k={kc:5d}: FAILED {str(e)[:110]}")
