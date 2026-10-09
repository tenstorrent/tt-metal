# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed end-to-end prefill latency of the full 64-layer model per bucket, batch 1.

One real prompt per bucket: the demo urgency question (100 tokens, bucket 128) and decision-golden
rows s01 (178 -> 1024), m06 (2006 -> 2048), l05 (3906 -> 4096), x04 (7928 -> 8192).

Measured per bucket in two modes, median reported: ``sustained`` (``WARMUP`` untimed then ``TIMED``
passes back to back) and ``burst`` (``BURST_RUNS`` passes, each after ``BURST_PAUSE_S`` s idle).
The device runs slower under sustained back-to-back load (see doc/full_model/README.md). Targets:
- ``device_forward``: tokens already on device; ``synchronize -> t0 -> model(...) -> synchronize -> t1``
  (host dispatch + device execution of embedding, 64 layers and the decision head);
- ``request``: the whole ``TTDecider`` request: render + tokenize (``AppTokenizer``), right-pad and
  upload, forward, and the readback of the ``count`` probabilities (``ttnn.to_torch`` syncs);
- ``tokenize`` and ``upload`` alone, to split the host part of ``request``;
- ``traced_forward`` (``test_traced_forward``): the same forward captured once as a ttnn trace and
  replayed (inputs fixed: same tokens, last index and count), to size the host dispatch share.

The projection compares ``device_forward`` with the stage-2 per-layer medians
(48 x linear_attention + 16 x full_attention layer; doc/fused_decoder/README.md).

Results: ``$PPLX_DECIDER_STAGE6_DIR/model_perf.json``. Profile capture (one warmed full forward between
signposts, for tt-perf-report)::

    python -m tracy -r -p -v -m pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py -k test_profile_model
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import DEMO_STATE, TTDecider, demo_questions
from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
from models.demos.pplx_decider_v1_27b.tt.model import BUCKETS, PplxDeciderModel

WARMUP = 2
TIMED = 7
BURST_RUNS = 5
BURST_PAUSE_S = 10.0
OUT_DIR = Path(os.environ.get("PPLX_DECIDER_STAGE6_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage6"))
PROMPTS = (
    Path(
        os.environ.get("PPLX_DECIDER_DECISION_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/decisions")
    )
    / "prompts.jsonl"
)
BUCKET_ROWS = {
    1024: "s01_ticket_routing",
    2048: "m06_server_500",
    4096: "l05_league_leader",
    8192: "x04_customer_lookup",
}
# Stage-2 per-layer medians in ms (doc/fused_decoder/README.md, "after" column), per bucket.
STAGE2_LAYER_MS = {
    "linear_attention": {128: 2.37, 1024: 6.03, 2048: 10.64, 4096: 21.67, 8192: 45.23},
    "full_attention": {128: 2.18, 1024: 5.03, 2048: 9.26, 4096: 21.09, 8192: 55.20},
}
TRACE_REGION = 256 * 1024 * 1024

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": TRACE_REGION})

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


@pytest.fixture(scope="module")
def decider(_device_module_impl):
    return TTDecider(PplxDeciderModel.from_snapshot(_device_module_impl), AppTokenizer())


def bucket_rows(tokenizer: AppTokenizer) -> dict[int, dict]:
    rows = {128: {"state": DEMO_STATE, "question": demo_questions()["urgency"]}}
    by_id = {json.loads(line)["id"]: json.loads(line)["row"] for line in PROMPTS.read_text().splitlines()}
    rows.update({bucket: by_id[rid] for bucket, rid in BUCKET_ROWS.items()})
    for bucket, row in rows.items():
        n = len(tokenizer.input_ids(row))
        assert n <= bucket and (bucket == BUCKETS[0] or n > BUCKETS[BUCKETS.index(bucket) - 1]), (bucket, n)
    return rows


def timed(fn, device=None, *, warmup=WARMUP, runs=TIMED, pause_s=0.0) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        time.sleep(pause_s)
        if device is not None:
            ttnn.synchronize_device(device)
        start = time.perf_counter()
        fn()
        if device is not None:
            ttnn.synchronize_device(device)
        samples.append(time.perf_counter() - start)
    return samples


def stats(samples: list[float]) -> dict:
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
        "max_ms": max(samples) * 1e3,
        "samples_ms": [round(s * 1e3, 3) for s in samples],
    }


def _save(name: str, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")


@pytest.mark.timeout(3600)
def test_model_perf(decider):
    model, tok = decider.model, decider.tokenizer
    device = model.mesh_device
    results = {}
    for bucket, row in bucket_rows(tok).items():
        ids, count = tok.input_ids(row), tok.count(row)
        tokens, last_index = model.upload_tokens(ids)

        def forward():
            probs, logits, _ = model(tokens, last_index, count)
            ttnn.deallocate(probs)
            ttnn.deallocate(logits)

        def upload():
            ttnn.deallocate(model.upload_tokens(ids)[0])

        def request_fn():
            decider.predict_probabilities(row["state"], row["question"])

        # Sustained: passes back to back. Burst: BURST_PAUSE_S idle before each pass (device clocks
        # recover; measured 1.27-1.37x slower back to back at 1024-2048, see doc/full_model/README.md).
        device_forward = stats(timed(forward, device))
        request = stats(timed(request_fn))
        device_forward_burst = stats(timed(forward, device, runs=BURST_RUNS, pause_s=BURST_PAUSE_S))
        request_burst = stats(timed(request_fn, runs=BURST_RUNS, pause_s=BURST_PAUSE_S))
        tokenize = stats(timed(lambda: tok.input_ids(row)))
        upload_ = stats(timed(upload, device))
        projected = 48 * STAGE2_LAYER_MS["linear_attention"][bucket] + 16 * STAGE2_LAYER_MS["full_attention"][bucket]
        results[bucket] = {
            "seq_len": len(ids),
            "count": count,
            "device_forward_sustained": device_forward,
            "request_sustained": request,
            "device_forward_burst": device_forward_burst,
            "request_burst": request_burst,
            "tokenize": tokenize,
            "upload": upload_,
            "decisions_per_s_sustained": 1e3 / request["median_ms"],
            "projected_stage2_layers_ms": projected,
            "burst_vs_projected": device_forward_burst["median_ms"] / projected,
            "sustained_vs_projected": device_forward["median_ms"] / projected,
        }
        ttnn.deallocate(tokens)
        print(
            f"PERF bucket {bucket:<5} S={len(ids):<5} device_forward sustained {device_forward['median_ms']:8.2f} "
            f"burst {device_forward_burst['median_ms']:8.2f} ms | request sustained {request['median_ms']:8.2f} "
            f"burst {request_burst['median_ms']:8.2f} ms ({results[bucket]['decisions_per_s_sustained']:.2f} decisions/s "
            f"sustained) | tokenize {tokenize['median_ms']:.2f} upload {upload_['median_ms']:.2f} | "
            f"projected {projected:.1f} ms"
        )
    _save(
        "model_perf.json",
        {
            "method": (
                f"sustained: {WARMUP} warm-ups + median of {TIMED} back to back; burst: median of {BURST_RUNS} "
                f"passes each after {BURST_PAUSE_S} s idle; batch 1; eager; synchronize_device around device passes"
            ),
            "policy": model.config.optimizations.policy.name,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "buckets": results,
        },
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("bucket", [128, 1024, 2048])
def test_traced_forward(decider, bucket):
    """Capture the full forward once as a ttnn trace, replay it, compare time and output to eager."""
    model, tok = decider.model, decider.tokenizer
    device = model.mesh_device
    row = bucket_rows(tok)[bucket]
    ids, count = tok.input_ids(row), tok.count(row)
    tokens, last_index = model.upload_tokens(ids)

    eager_probs, eager_logits, _ = model(tokens, last_index, count)  # compile
    eager = ttnn.to_torch(eager_probs)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    probs, logits, _ = model(tokens, last_index, count)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    def replay():
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

    def eager_forward():
        p, l, _ = model(tokens, last_index, count)
        ttnn.deallocate(p)
        ttnn.deallocate(l)

    traced = stats(timed(replay, device))
    eager_t = stats(timed(eager_forward, device))
    replayed = ttnn.to_torch(probs)
    ttnn.release_trace(device, trace_id)
    result = {
        "bucket": bucket,
        "seq_len": len(ids),
        "eager_device_forward": eager_t,
        "traced_device_forward": traced,
        "speedup": eager_t["median_ms"] / traced["median_ms"],
        "trace_output_bit_identical_to_eager": bool((replayed == eager).all()),
        "note": "trace bakes last_index/count/tokens; a serving trace would need them as device inputs",
    }
    _save(f"trace_bucket{bucket}.json", result)
    print(
        f"TRACE bucket {bucket}: eager {eager_t['median_ms']:.2f} ms, traced {traced['median_ms']:.2f} ms, "
        f"speedup {result['speedup']:.3f}x, identical {result['trace_output_bit_identical_to_eager']}"
    )
    for t in (tokens, eager_probs, eager_logits, probs, logits):
        ttnn.deallocate(t)
    assert result["trace_output_bit_identical_to_eager"]


@pytest.mark.timeout(3600)
def test_profile_model(decider):
    """One warmed full forward at the 2048 bucket between signposts (device-profiler capture)."""
    model, tok = decider.model, decider.tokenizer
    row = bucket_rows(tok)[2048]
    ids, count = tok.input_ids(row), tok.count(row)
    tokens, last_index = model.upload_tokens(ids)
    # ~1.7k programs per forward: flush the device profiler buffers after load and after every warm-up
    # (run with TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 so one forward fits).
    ttnn.ReadDeviceProfiler(model.mesh_device)
    for _ in range(WARMUP):
        probs, logits, _ = model(tokens, last_index, count)
        ttnn.synchronize_device(model.mesh_device)
        ttnn.ReadDeviceProfiler(model.mesh_device)
    signpost("PREFILL_START")
    probs, logits, _ = model(tokens, last_index, count)
    ttnn.synchronize_device(model.mesh_device)
    signpost("PREFILL_END")
