# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command 3 contract: host-free full pipeline + per-stage trace capture."""

from __future__ import annotations

import pytest
import torch

from ...tt.pipeline import PIPELINE_STAGES, build_pipeline
from . import _golden


def _model():
    from transformers import Qwen2VLForConditionalGeneration

    m = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    m.eval()
    return m


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_host_op_selftest(device):
    """The full model math (vision prefix + text stages) fires ZERO host aten ops."""
    pipe = build_pipeline(device, _model())
    v = pipe.host_op_selftest()
    assert v["on_device"], f"host aten ops fired in forward: {v['host_ops']}"


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 200000000, "num_command_queues": 2}], indirect=True
)
def test_trace_capture_selftest(device):
    """Every stage in PIPELINE_STAGES captures host-free into a trace and replays."""
    g = _golden()
    pipe = build_pipeline(device, _model())
    ref = {"prefill": g["man_logits"][0].float()}
    ok, results = pipe.trace_capture_selftest(device, reference_logits=ref)
    print(f"PIPELINE_STAGES={PIPELINE_STAGES}")
    print(f"trace_capture results={results}")
    assert ok, f"a stage failed host-free trace capture: {results}"
    if results.get("prefill", {}).get("pcc") is not None:
        assert results["prefill"]["pcc"] >= 0.90, results["prefill"]
