# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage trace capture for Call 1, and the `trace_caps` sidecar the trace gate reads.

    ./python_env/bin/python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -s

For each stage in `PIPELINE_STAGES` this captures ONE host-op-free step at a pinned shape,
executes the trace, checks it against the eager output, and RELEASES it before the next stage
(stage traces must not co-reside). The measurement object comes from `build_pipeline`, the same
single build surface the demo and the e2e test use.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import ttnn

from models.demos.voxtral_tts_full.tt.pipeline import (
    PIPELINE_STAGES,
    TRACE_FRAME_CAPACITY,
    TRACE_REGION_SIZE,
    TRACE_SEQ_CAPACITY,
    build_pipeline,
    pcc,
)

_SIDECAR = Path(__file__).with_suffix(".py.trace_caps.json")
_PCC_GATE = 0.99


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def pipe(device):
    return build_pipeline(device)


def test_trace_capture_per_stage(pipe, device):
    """Capture, execute and verify one traced step per stage; write the trace_caps sidecar."""
    stages, engaged = {}, True

    for stage in PIPELINE_STAGES:
        capacity = TRACE_FRAME_CAPACITY if stage == "vocode" else TRACE_SEQ_CAPACITY
        record = {"captured": False, "capacity": capacity, "pcc": None, "ms": None}

        while True:
            inputs = getattr(pipe, f"{stage}_trace_inputs")()
            inputs["capacity"] = capacity
            getattr(pipe, f"{stage}_trace_setup")(inputs)
            step = getattr(pipe, f"{stage}_trace_step")

            reference = ttnn.to_torch(step()).float()  # eager, outside the trace
            try:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = step()
                ttnn.end_trace_capture(device, tid, cq_id=0)
            except RuntimeError as exc:
                if capacity > 32:
                    capacity //= 2
                    print(f"[trace] {stage}: capture overflowed the region -> shrinking C to {capacity} ({exc})")
                    record["capacity"] = capacity
                    continue
                print(f"[trace] {stage}: capture FAILED at C={capacity}: {exc}")
                engaged = False
                break

            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            t0 = time.time()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            record["ms"] = (time.time() - t0) * 1e3
            record["pcc"] = pcc(ttnn.to_torch(out).float(), reference)
            record["captured"] = True
            ttnn.release_trace(device, tid)
            print(f"[trace] {stage}: C={capacity} PCC={record['pcc']:.6f} execute_trace {record['ms']:.2f} ms")
            break

        engaged = engaged and record["captured"] and (record["pcc"] or 0) >= _PCC_GATE
        stages[stage] = record

    _SIDECAR.write_text(
        json.dumps(
            {
                "trace_1cq": bool(engaged),
                "stages": stages,
                "trace_region_size": TRACE_REGION_SIZE,
                "pipeline_stages": list(PIPELINE_STAGES),
            },
            indent=2,
        )
    )
    print(f"[trace] wrote {_SIDECAR}")

    for stage, record in stages.items():
        assert record["captured"], f"{stage}: trace did not capture"
        assert record["pcc"] >= _PCC_GATE, f"{stage}: traced output PCC {record['pcc']} < {_PCC_GATE}"


def test_trace_capture_selftest(pipe, device):
    """The model-agnostic recipe, through the pipeline's own entry point."""
    assert pipe.trace_capture_selftest(device)


def test_host_op_selftest(pipe):
    verdict = pipe.host_op_selftest(max_frames=2)
    print(f"[host-op] {verdict['reason']}")
    assert verdict["on_device"], verdict["host_ops"]
