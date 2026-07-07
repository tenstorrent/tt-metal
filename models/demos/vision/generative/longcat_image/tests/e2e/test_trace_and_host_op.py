# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command 3 — trace+2CQ contract + host-op observer gate for LongCat-Image.

Two on-device checks over the shared pipeline (tt/pipeline.py), derived from the
HF reference config's PIPELINE_STAGES = [text_encode, denoise, vae_decode]:

  * test_host_op_selftest       — the AUTHORITATIVE fully-on-device check: each
    stage's model math runs under host_op_observer.observe_host_ops() with input
    encoding + weight build + constant precompute OUTSIDE the observed region.
    A truly on-device forward fires ZERO host aten ops.

  * test_trace_capture_selftest — captures ONE host-free trace_step per stage in
    ttnn.begin/end_trace_capture, execute_trace, PCC-checks vs the eager step,
    then releases before the next stage (stage traces never co-reside).
"""

from __future__ import annotations

import os

import pytest
import torch

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
SELFTEST_MAXLEN = int(os.environ.get("LONGCAT_SELFTEST_MAXLEN", "32"))
SELFTEST_SIZE = int(os.environ.get("LONGCAT_SELFTEST_SIZE", "128"))


@pytest.fixture(scope="module")
def pipe():
    from diffusers import LongCatImagePipeline

    print(f"[c3] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    p = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    p.set_progress_bar_config(disable=True)
    return p


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_host_op_selftest(device_params, device, pipe):
    from models.demos.vision.generative.longcat_image.tt import pipeline as P

    ttp = P.LongCatImagePipelineTT(device, pipe)
    verdict = ttp.host_op_selftest(max_length=SELFTEST_MAXLEN, size=SELFTEST_SIZE)
    for stage, v in verdict["per_stage"].items():
        print(f"[c3] host_op {stage}: on_device={v['on_device']} host_ops={v['host_ops'][:8]}", flush=True)
    assert verdict["on_device"], f"host_op_selftest FAIL — host aten ops fired: {verdict['host_ops'][:16]}"


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 400000000}], indirect=True
)
def test_trace_capture_selftest(device_params, device, pipe):
    from models.demos.vision.generative.longcat_image.tt import pipeline as P

    ttp = P.LongCatImagePipelineTT(device, pipe)
    ok = ttp.trace_capture_selftest(device, max_length=SELFTEST_MAXLEN, size=SELFTEST_SIZE)
    assert ok, "trace_capture_selftest — one or more stages did not capture host-free with a PCC match (see log)"
