# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Everything-on-device gate: the pipeline must be trace-capturable per stage,
and its forward must fire no host tensor op.

`test_trace_capture_per_stage` runs the pipeline's own
`trace_capture_selftest`: for EACH entry in PIPELINE_STAGES it pins the stage's
inputs, wraps ONE step in `ttnn.begin_trace_capture` / `end_trace_capture`,
replays it with `ttnn.execute_trace` and checks the replayed output against the
eager step before releasing the trace. A host op inside the step makes the
capture raise, so this failing is the signal that something left the device.

`test_forward_fires_no_host_op` runs both task heads' forwards inside
`host_op_observer.observe_host_ops()` (a TorchDispatchMode): tokenization, the
one-time weight build and the readback stay outside the region, so any aten op
seen inside it is host compute on the critical path.

The measured capture is recorded next to this file as the trace-capability
sidecar the downstream trace gate reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.demos.voxtral_tts_backbone.selftest_device import TRACE_REGION_SIZE
from models.demos.voxtral_tts_backbone.tt.pipeline import TRACE_REPORT, host_op_selftest, trace_capture_selftest

CAPS_PATH = Path(__file__).resolve().parent / (Path(__file__).name + ".trace_caps.json")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}], indirect=True
)
def test_trace_capture_per_stage(device):
    captured = trace_capture_selftest(device)
    report = dict(TRACE_REPORT)
    stages = report.get("stages") or {}
    print("\n[trace] %s" % json.dumps(report, indent=2), flush=True)
    caps = {
        "trace_1cq": bool(captured) and bool(stages),
        "trace_1cq_path": "models/demos/voxtral_tts_backbone/tt/pipeline.py::trace_capture_selftest",
        "eager_terminal": False,
        "measured_by": "models/demos/voxtral_tts_backbone/tests/e2e/test_trace_capture.py",
        "stages": stages,
    }
    CAPS_PATH.write_text(json.dumps(caps, indent=2))
    assert stages, "no stage was captured: PIPELINE_STAGES produced nothing to trace"
    for stage, detail in stages.items():
        assert detail["captured"], "stage %s did not capture" % stage
        assert detail["replay_matches_eager"], "stage %s replay disagrees with eager (corr %s)" % (
            stage,
            detail["replay_corr"],
        )
    assert captured, "trace_capture_selftest reported a stage failure: %s" % report


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forward_fires_no_host_op(device):
    verdict = host_op_selftest(device)
    print("\n[host-op] %s" % json.dumps(verdict, indent=2), flush=True)
    assert verdict["on_device"], "the forward fired host aten ops: %s" % verdict["host_ops"]
