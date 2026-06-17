# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Trace-captured perf microbenchmark for MoE1D (M5).

Measures device time for a single MoE1D forward, per (family x mode), on a 1D mesh. Builds the module
+ persistent inputs once, warms up (compile + lazy weight load), trace-captures the forward, then
replays it in a signposted loop so host dispatch is removed from the measurement. Reports median + range
of per-forward time over the replayed iterations.

Boundary equivalence (Step 0 of the perf skill): MoE1D's forward IS the same op sequence the Gemma4 /
GPT-OSS routed-expert references run — router (rms_norm/linear/softmax/topk/scatter) + per-expert
`sparse_matmul` gate/up/down + gated activation + routing-weighted reduce (+ TP all-reduce on >1 device).
At matched config (same E/H/I/top_k/dtype/mesh) the two are the same workload, so this self-baseline is
the reference parity point; a direct side-by-side timing of the reference modules is a documented
follow-up (it requires standing up their full ccl_manager/mesh_config/hf_config machinery).

On a single device there are no collectives, so device time is entirely compute (matmul + eltwise); the
compute/collective split is meaningful only on >1 device (TP all-reduce), deferred with the multi-device
bringup.

Run via the device queue, e.g.:
    pytest models/common/tests/modules/moe/test_moe_1d_perf.py -rA -s
For the per-op compute breakdown, run the same node under the device profiler
(TT_METAL_DEVICE_PROFILER=1) and read the ops log between the "start"/"stop" signposts.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoE1DConfig, RoutingStrategy

try:
    from tracy import signpost
except ImportError:  # tracy not always importable outside the profiler build

    def signpost(_msg):  # noqa: D103
        pass


# Representative dims (match the functional test; small for fast, stable iteration).
H = 256
I = 128
E = 8
TOP_K = 2
NUM_ITERS = 30


def _build_module(family, mesh_device):
    torch.manual_seed(0)
    gate_w = torch.randn(1, E, H, I) * 0.1
    up_w = torch.randn(1, E, H, I) * 0.1
    down_w = torch.randn(1, E, I, H) * 0.1
    router_w = torch.randn(1, 1, H, E)

    def lz(src, dt=ttnn.bfloat8_b):
        return LazyWeight(source=src, dtype=dt)

    if family == "gemma4":
        cfg = MoE1DConfig(
            gate_proj=lz(gate_w),
            up_proj=lz(up_w),
            down_proj=lz(down_w),
            router_weight=lz(router_w, ttnn.bfloat16),
            top_k=TOP_K,
            routing_strategy=RoutingStrategy.SOFTMAX_TOPK_SUMNORM,
            activation_strategy=ExpertActivation.GEGLU,
            router_prenorm_eps=1e-6,
            router_input_scalar=H**-0.5,
        )
    else:
        gb = torch.randn(1, E, I) * 0.05
        ub = torch.randn(1, E, I) * 0.05
        db = torch.randn(1, E, H) * 0.05
        rb = torch.randn(1, 1, 1, E) * 0.1
        cfg = MoE1DConfig(
            gate_proj=lz(gate_w),
            up_proj=lz(up_w),
            down_proj=lz(down_w),
            router_weight=lz(router_w, ttnn.bfloat16),
            top_k=TOP_K,
            routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
            activation_strategy=ExpertActivation.SWIGLU_CLAMP,
            swiglu_limit=7.0,
            swiglu_alpha=1.702,
            router_bias=lz(rb, ttnn.bfloat16),
            gate_bias=lz(gb, ttnn.bfloat16),
            up_bias=lz(ub, ttnn.bfloat16),
            down_bias=lz(db, ttnn.bfloat16),
        )
    return MoE1D.from_config(cfg)


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("mode,seq_len", [("decode", 1), ("prefill", 128)], ids=["decode", "prefill-128"])
@pytest.mark.parametrize("family", ["gemma4", "gptoss"])
def test_moe_1d_perf(ttnn_mesh_device: ttnn.MeshDevice, family, mode, seq_len):
    mesh_device = ttnn_mesh_device
    ttnn.SetDefaultDevice(mesh_device)
    moe = _build_module(family, mesh_device)

    router_in = ttnn.from_torch(
        torch.randn(1, 1, seq_len, H) * 0.5,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    expert_in = ttnn.from_torch(
        torch.randn(1, 1, seq_len, H) * 0.5,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # 1. Warm-up / compile (also triggers lazy weight load + prefill-sparsity cache).
    _ = moe.forward(router_in, expert_in, mode)
    ttnn.synchronize_device(mesh_device)

    # 2. Capture the trace (host dispatch happens once here).
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    _ = moe.forward(router_in, expert_in, mode)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # 3. Replay between signposts — this is the measured (device-bound) region.
    #    Per-iter sync gives a per-forward device time sample (median + range below).
    samples_us = []
    signpost("start")
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_us.append((time.perf_counter() - t0) * 1e6)
    signpost("stop")

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.SetDefaultDevice(None)

    samples_us.sort()
    median = samples_us[len(samples_us) // 2]
    lo, hi = samples_us[0], samples_us[-1]
    logger.info(
        f"MoE1D PERF [{family}/{mode} seq={seq_len} (1,1)]: "
        f"median={median:.1f}us range=[{lo:.1f},{hi:.1f}]us over {NUM_ITERS} iters "
        f"(compute-only; no collectives on 1 device)"
    )
    assert median > 0
