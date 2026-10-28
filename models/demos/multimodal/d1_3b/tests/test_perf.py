# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced prefill latency per bucket on a single Wormhole chip.

    MESH_DEVICE=N150 pytest models/demos/multimodal/d1_3b/tests/test_perf.py -s

Reports device-only prefill (trace replay + readback) per bucket and end-to-end latency (render, tokenize,
option readout included) for a short question. Numbers are measured, not promised; vendor anchors for one
short question are 8 ms (RTX 4090, bf16, CUDA graphs) and 50 ms (Jetson Orin Nano).
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest

from models.demos.multimodal.d1_3b.tt.d1 import TtD1
from models.demos.multimodal.d1_3b.tt.decision import D1Prompt
from models.demos.multimodal.d1_3b.tt.model import L1_SMALL_SIZE, PREFILL_BUCKETS, TRACE_REGION_SIZE, TtD1Model

ITERS = 20
WARMUP = 3
RESULTS_PATH = Path(os.environ.get("D1_RESULTS_DIR", "generated/d1_3b")) / "perf.json"
SHORT_STATE = "I was charged twice this month, please refund one of them."
SHORT_QUESTION = {"type": "noul", "instructions": "Is the customer asking for a refund?"}


def _p(times, q):
    return statistics.quantiles(times, n=100)[q - 1] if len(times) >= 2 else times[0]


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_prefill_perf(mesh_device):
    from loguru import logger

    model = TtD1Model(mesh_device)
    model.capture_traces()

    # Device-only prefill per bucket: a bucket-filling id list, timed around hidden().
    base = model.args.tokenizer.encode(SHORT_STATE, add_special_tokens=False)
    buckets = []
    for size in PREFILL_BUCKETS:
        if size > model.args.max_seq_len:
            continue
        ids = (base * (size // len(base) + 1))[:size]
        for _ in range(WARMUP):
            model.hidden(ids)
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            model.hidden(ids)
            times.append((time.perf_counter() - t0) * 1000)
        row = {"bucket": size, "p50_ms": _p(times, 50), "p90_ms": _p(times, 90), "min_ms": min(times)}
        buckets.append(row)
        logger.info(
            f"bucket {size:>5}: p50 {row['p50_ms']:.1f} ms  p90 {row['p90_ms']:.1f} ms  min {row['min_ms']:.1f} ms"
        )

    # End-to-end one short question: render + tokenize + prefill + readout.
    d1 = TtD1(model, D1Prompt(tokenizer=model.args.tokenizer))
    for _ in range(WARMUP):
        d1.system_one(SHORT_STATE, {"q": SHORT_QUESTION})
    e2e = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        d1.system_one(SHORT_STATE, {"q": SHORT_QUESTION})
        e2e.append((time.perf_counter() - t0) * 1000)
    tokens = len(d1.prompt.input_ids(SHORT_STATE, SHORT_QUESTION))
    summary = {
        "short_question_tokens": tokens,
        "e2e_p50_ms": _p(e2e, 50),
        "e2e_p90_ms": _p(e2e, 90),
        "e2e_min_ms": min(e2e),
        "buckets": buckets,
    }
    logger.info(
        f"end-to-end 1 short question ({tokens} tok): p50 {summary['e2e_p50_ms']:.1f} ms  min {summary['e2e_min_ms']:.1f} ms"
    )
    model.release_traces()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=1) + "\n")
