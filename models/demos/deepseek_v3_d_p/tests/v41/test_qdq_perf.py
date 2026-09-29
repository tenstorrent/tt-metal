# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Traced per-call time of the V4.1 QDQ ops (tt/v41/qdq.py) at production per-chip shapes (bead 8y7.9.4).

One chip, chunk 5120 on the 4x2 mesh (sp = 4, tp = 2): 1280 rows per chip. Each case captures one call in a
trace and replays it; the reported time is the replay wall time per call (device synchronized), which is what
the traced model pays. The G2 bound is the DRAM roofline of one bf16 read + one bf16 write of the tensor.

Logged as ``V41_QDQ_PERF`` JSON lines, also appended to ``$V41_QDQ_PERF_OUT`` (default
``generated/v41_qdq_perf.log``); no pass/fail bar (the bead compares before / after against G2).
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41 import qdq

DRAM_GBPS = 512.0  # Blackhole p150 DRAM peak (G2 bound)
REPLAYS = 20

# case id -> (qdq function name, per-chip shape)
CASES = {
    "fp8_hidden": ("fp8_qdq", (1, 1, 1280, 5120)),
    "fp8_q_lora": ("fp8_qdq", (1, 1, 1280, 1280)),
    "fp8_window_kv": ("fp8_qdq", (1, 1, 1280, 512)),
    "fp4_ue8m0_index_q": ("fp4_ue8m0_qdq", (1, 32, 640, 128)),
    "fp4_ue8m0_index_k": ("fp4_ue8m0_qdq", (1, 1, 1280, 128)),
    "fp4_e4m3_comp_kv": ("fp4_e4m3_qdq", (1, 1, 640, 512)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 << 20}], indirect=True)
def test_qdq_traced_time(device, case):
    name, shape = CASES[case]
    fn = getattr(qdq, name)
    x = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    t0 = time.perf_counter()
    fn(x)  # compile + program cache
    ttnn.synchronize_device(device)
    compile_s = time.perf_counter() - t0

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn(x)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm replay
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per_call_us = (time.perf_counter() - t0) / REPLAYS * 1e6
    ttnn.release_trace(device, tid)
    out.deallocate()

    numel = 1
    for d in shape:
        numel *= d
    bound_us = 4 * numel / (DRAM_GBPS * 1e3)
    record = {
        "case": case,
        "fn": name,
        "shape": list(shape),
        "traced_us": round(per_call_us, 1),
        "dram_bound_us": round(bound_us, 1),
        "g2_target_us": round(2 * bound_us, 1),
        "dram_util": round(bound_us / per_call_us, 3),
        "first_call_s": round(compile_s, 2),
    }
    line = "V41_QDQ_PERF " + json.dumps(record)
    logger.info(line)
    with open(os.environ.get("V41_QDQ_PERF_OUT", "generated/v41_qdq_perf.log"), "a") as f:
        f.write(line + "\n")
