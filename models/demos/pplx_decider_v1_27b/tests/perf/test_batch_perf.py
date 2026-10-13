# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched request latency and decisions/s vs batch size, buckets 1024 and 2048 (stage-6 harness style).

Per bucket and batch size B (1, 2, 4, 8 and the bucket's ``BATCH_CAPS`` entry), the rows are the decision
golden prompts of that bucket (10 at 1024, 6 at 2048), cycled to B. Measured, median reported:
- ``device_forward`` sustained: tokens already on device, ``synchronize -> forward -> synchronize``,
  ``WARMUP`` untimed then ``TIMED`` back to back;
- ``request`` sustained and burst: ``TTDecider.predict_probabilities_batch`` (render + tokenize, pad,
  upload, forward, one readback), back to back and each pass after ``BURST_PAUSE_S`` s idle.
B = 1 runs the batch-1 path (``predict_probabilities`` / int last index), as the stage-6 harness.
Per-request latency is the latency of the batch (every request of a batch waits for the whole batch);
decisions/s = B / batch latency.

Results: ``$PPLX_DECIDER_STAGE6_DIR/batch_perf.json``.

    pytest models/demos/pplx_decider_v1_27b/tests/perf/test_batch_perf.py -q -s
"""

from __future__ import annotations

import json
import time

import pytest

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
from models.demos.pplx_decider_v1_27b.tests.perf.test_model_perf import stats, timed
from models.demos.pplx_decider_v1_27b.tt import paths
from models.demos.pplx_decider_v1_27b.tt.model import BATCH_CAPS, PplxDeciderModel, bucket_for

BUCKETS = (1024, 2048)
SIZES = (1, 2, 4, 8)
WARMUP, TIMED = 2, 5
BURST_RUNS, BURST_PAUSE_S = 3, 10.0
OUT_DIR = paths.E2E_DIR

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})


@pytest.fixture(scope="module")
def decider(_device_module_impl):
    model = PplxDeciderModel.from_snapshot(_device_module_impl, max_batch_size=max(BATCH_CAPS.values()))
    return TTDecider(model, AppTokenizer())


@pytest.mark.timeout(7200)
def test_batch_perf(decider):
    model, tok = decider.model, decider.tokenizer
    device = model.mesh_device
    prompts = [
        json.loads(l)
        for l in paths.require(paths.DECISION_GOLDEN_DIR / "prompts.jsonl", "decision").read_text().splitlines()
    ]
    results = {}
    for bucket in BUCKETS:
        pool = [p["row"] for p in prompts if bucket_for(p["seq_len"]) == bucket]
        for size in sorted(set(SIZES) | {BATCH_CAPS[bucket]}):
            rows = [pool[i % len(pool)] for i in range(size)]
            ids = [tok.input_ids(r) for r in rows]
            counts = [tok.count(r) for r in rows]
            assert {bucket_for(len(i)) for i in ids} == {bucket}
            if size == 1:
                tokens, last = model.upload_tokens(ids[0])
                count = counts[0]

                def request():
                    decider.predict_probabilities(rows[0]["state"], rows[0]["question"])

            else:
                tokens, last = model.upload_batch(ids)
                count = counts

                def request():
                    assert len(decider.plan_batches(rows)[2]) == 1  # one forward
                    decider.predict_probabilities_batch(rows)

            def forward():
                probs, logits, _ = model(tokens, last, count)
                ttnn.deallocate(probs)
                ttnn.deallocate(logits)

            fwd = stats(timed(forward, device, warmup=WARMUP, runs=TIMED))
            req = stats(timed(request, warmup=1, runs=TIMED))
            req_burst = stats(timed(request, warmup=0, runs=BURST_RUNS, pause_s=BURST_PAUSE_S))
            ttnn.deallocate(tokens)
            rec = {
                "batch": size,
                "device_forward_sustained": fwd,
                "request_sustained": req,
                "request_burst": req_burst,
                "decisions_per_s_sustained": size * 1e3 / req["median_ms"],
                "decisions_per_s_burst": size * 1e3 / req_burst["median_ms"],
                "device_decisions_per_s": size * 1e3 / fwd["median_ms"],
            }
            results.setdefault(str(bucket), {})[str(size)] = rec
            print(
                f"BATCHPERF bucket {bucket} B={size:<3} forward {fwd['median_ms']:8.1f} ms | request sustained "
                f"{req['median_ms']:8.1f} burst {req_burst['median_ms']:8.1f} ms | decisions/s sustained "
                f"{rec['decisions_per_s_sustained']:.2f} burst {rec['decisions_per_s_burst']:.2f} "
                f"device {rec['device_decisions_per_s']:.2f}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "batch_perf.json").write_text(
        json.dumps(
            {
                "method": (
                    f"device_forward: {WARMUP} warm-ups + median of {TIMED} back to back; request: 1 warm-up + "
                    f"median of {TIMED} back to back (sustained) and median of {BURST_RUNS} passes each after "
                    f"{BURST_PAUSE_S} s idle (burst); eager; rows = decision golden prompts of the bucket, cycled"
                ),
                "policy": model.config.optimizations.policy.name,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "buckets": results,
            },
            indent=2,
        )
        + "\n"
    )
