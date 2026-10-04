# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the MLA projections cost when TP sharding goes away.

Under the SP-batch scheme each device holds the full hidden/head dim instead of a quarter, so the three
TP-split projections widen 4x per device. mla.py splits them two ways (activation is
[1, 1, seq/sp, hidden/tp]):

  q_a_proj  row-parallel    K x4  (1536 -> 6144), followed by a TP reduce-scatter
  q_b_proj  column-parallel N x4  (4096 -> 16384), no collective
  o_proj    row-parallel    K x4  (4096 -> 16384), followed by a TP reduce-scatter

GLM-5.2: hidden 6144, q_lora 2048, 64 heads x (192 nope + 64 rope) = 16384, v_head_dim 256.
M = chunk/8 -- the sequence is sharded over SP whether or not TP is in play.
"""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SP, TP = 8, 4
HIDDEN, Q_LORA, HEADS, QK_HEAD, V_HEAD = 6144, 2048, 64, 192 + 64, 256
ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
OUT = Path(os.environ.get("GLM_MM_SWEEP_OUT", "generated/glm_mm"))

# name, K at tp=4, K at tp=1, N at tp=4, N at tp=1, parallelism
CASES = [
    ("q_a_proj", HIDDEN // TP, HIDDEN, Q_LORA, Q_LORA, "row (K sharded)"),
    ("q_b_proj", Q_LORA, Q_LORA, HEADS * QK_HEAD // TP, HEADS * QK_HEAD, "column (N sharded)"),
    ("o_proj", HEADS * V_HEAD // TP, HEADS * V_HEAD, HIDDEN, HIDDEN, "row (K sharded)"),
]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((SP, TP), torus_xy_device_params(trace_region_size=8 * 1024 * 1024), id="8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk", [5120, 2048], ids=["chunk5120", "chunk2048"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("tp", [4, 1], ids=["tp4", "tp1"])
@pytest.mark.timeout(600)
def test_mla_projection_width(mesh_device, device_params, chunk, case, tp):
    name, k4, k1, n4, n1, kind = case
    K, N = (k4, n4) if tp == 4 else (k1, n1)
    M = chunk // SP
    torch.manual_seed(1234)
    mesh_device.enable_program_cache()

    def up(t, dt):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    act = up(torch.randn(1, 1, M, K), ttnn.bfloat16)
    w = up(torch.randn(1, 1, K, N), ttnn.bfloat8_b)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        return ttnn.linear(act, w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    try:
        warm = run_once()
    except RuntimeError as e:
        pytest.skip(f"rejected: {str(e).splitlines()[0][:200]}")
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = run_once()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def replay():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    durations = []
    try:
        profile_realtime_program_merged(mesh_device, replay, record_timeout_seconds=30.0)
        for _ in range(ITERS):
            _, programs = profile_realtime_program_merged(mesh_device, replay, record_timeout_seconds=30.0)
            durations.append(max(p["duration_ns"] for p in programs.values()))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(out)

    med = statistics.median(durations)
    flops = 2 * M * K * N
    rec = {
        "matmul": name,
        "kind": kind,
        "chunk": chunk,
        "tp": tp,
        "M": M,
        "K": K,
        "N": N,
        "median_us": round(med / 1000, 2),
        "gflop": round(flops / 1e9, 2),
        "tflops": round(flops / med * 1e-3, 2),
        "fpu_util_pct": round(flops / med * 1e-3 / (110 * 2048 * 1.35e-3) * 100, 2),
    }
    logger.info(f"mla projection: {rec}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rt_results.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    for t in (act, w):
        ttnn.deallocate(t)
