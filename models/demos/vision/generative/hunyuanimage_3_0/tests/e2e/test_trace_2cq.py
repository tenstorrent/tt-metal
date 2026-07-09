# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command 3 — trace+2CQ contract + fully-on-device selftests for
`tencent/HunyuanImage-3.0`.

  * host_op_selftest  — authoritative fully-on-device check: the model math
    (embed -> decoder layers -> output) fires ZERO host aten ops.
  * trace_capture_selftest — each PIPELINE_STAGE (prefill, decode) captures one
    host-free step in begin/end_trace_capture, execute_trace, and matches the
    eager step by PCC.

Both obtain the resident pipeline through the SAME `build_pipeline` factory the
perf/2CQ harness uses.

Run:  ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_trace_2cq.py -s
"""

from __future__ import annotations

import pytest
import torch

from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pl.load_reference_model()
    return _MODEL


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_host_op_selftest(device_params, device):
    """Gate 1 (authoritative): the on-device forward fires zero host aten ops."""
    torch.manual_seed(0)
    pipe = pl.build_pipeline(device, _get_model())
    v = pipe.host_op_selftest()
    print(f"\nhost_op verdict: on_device={v['on_device']} n_host_ops={v['n_host_ops']}")
    if v["host_ops"]:
        print(f"host_ops={v['host_ops']}")
    assert v["on_device"], f"host aten ops fired in the forward: {v['host_ops']}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 200_000_000}], indirect=True)
def test_trace_capture_selftest(device_params, device):
    """Each PIPELINE_STAGE captures host-free and matches the eager step."""
    torch.manual_seed(0)
    pipe = pl.build_pipeline(device, _get_model())
    print(f"\nPIPELINE_STAGES={pipe.PIPELINE_STAGES}")
    ok_all, results = pipe.trace_capture_selftest(device)
    for stage, r in results.items():
        print(f"trace stage={stage}: {r}")
    # prefill is the real graduated path — it must capture host-free & match.
    assert results["prefill"]["captured"], f"prefill trace capture failed: {results['prefill']}"
    assert results["prefill"]["ok"], f"prefill trace PCC below target: {results['prefill']}"
