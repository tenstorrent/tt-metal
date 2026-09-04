# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end cost of one Kimi-K3 KDA layer at the production shape.

The inversion inside `prepare_chunk_recurrence` went from 8 tile matmuls to 30 to make it correct at
Kimi-K3's real decay magnitudes, and the op's own microbenchmark on its smallest shape
(h2-n4-k32-v64) shows that as +42%. That shape is not what the model runs, and a 32x32 tile matmul
is not what a KDA layer spends its time on — the projections are 7168x12288. This measures the
figure that actually matters: one full `ttKDA.forward` at 5120 tokens on the 8x4 mesh.
"""
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kda_layer1_golden import PLACEMENTS, SEQ_LEN, _shard
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict

SP_AXIS, TP_AXIS = 0, 1
ITERATIONS = 20


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_kda_layer_forward_cost(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, 2)
    attention = build_attention(
        mesh_device,
        kimi_k3_hf_config(max_seq=SEQ_LEN),
        KimiK3Config,
        load_layer_state_dict(Path(checkpoint), 1),
        layer_idx=1,
        schedule=schedule,
        seq_len=SEQ_LEN,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )
    states = KdaStateCache({1: attention.kda})
    attention.bind_state_cache(states)
    hidden = _shard(mesh_device, trace.rows("kda", "kda_input_layer_0", 0, SEQ_LEN))

    try:
        for _ in range(3):  # warm the program cache
            ttnn.deallocate(attention.forward(hidden, K3AttnContext()))
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            ttnn.deallocate(attention.forward(hidden, K3AttnContext()))
        ttnn.synchronize_device(mesh_device)
        elapsed = (time.perf_counter() - start) / ITERATIONS
    finally:
        states.deallocate()

    logger.info(f"  ttKDA.forward 5120 tokens, 8x4: {elapsed * 1e3:.2f} ms/call over {ITERATIONS} iterations")
