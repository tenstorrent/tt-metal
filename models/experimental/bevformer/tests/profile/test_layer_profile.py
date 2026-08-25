# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for a single BEVFormer encoder layer.

Same shape as ``test_bevformer_encoder_profile``, one layer instead of six: PCC
gate, warmup, then signposted iterations so the report covers already-compiled,
already-dispatched programs. The reference points and the camera projection are
built once outside the measured region — the encoder does the same, so what is
measured here is the per-layer cost the encoder repeats.

No trace capture. ``TTSpatialCrossAttention.forward`` calls ``ttnn.to_torch`` on
``bev_mask`` and on its rebatch indices, and the host result decides shapes for
the ops that follow; reads and writes both TT_FATAL inside a capture region.
Until that host round-trip is gone the signposted region carries host dispatch,
so read it as end-to-end device time, not as a traced-replay figure.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.tests.layer_common import build_layer_fixture
from models.experimental.bevformer.tests.test_utils import check_with_pcc

PERF_WARMUP_ITERS = 1
DEVICE_PERF_ITERS = 1


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("config_name", ["nuscenes_base"])
@pytest.mark.parametrize("bev_size", [(100, 100)])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("expected_pcc", [0.997])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_layer_profile(
    device,
    config_name,
    bev_size,
    batch_size,
    expected_pcc,
    reset_seeds,
    ensure_gc,
):
    logger.info(f"device-perf run of commit {_head_sha()}")

    config = get_preset_config(config_name)
    assert config is not None, f"Configuration '{config_name}' not found"

    fixture = build_layer_fixture(device, config, bev_size, batch_size)

    ref_output = fixture.ref_model(**fixture.ref_inputs)

    def op_fn():
        return fixture.tt_model(**fixture.tt_inputs)

    tt_output = op_fn()
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)
    passed, message = check_with_pcc(ref_output, tt_output_torch, expected_pcc)
    assert passed, f"PCC check failed: {message}"
    logger.info(f"PCC gate: {message}")
    ttnn.deallocate(tt_output)

    for _ in range(PERF_WARMUP_ITERS):
        out = op_fn()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    ttnn.synchronize_device(device)
    outputs = []
    signpost("start")
    for _ in range(DEVICE_PERF_ITERS):
        outputs.append(op_fn())
        ttnn.synchronize_device(device)
    signpost("stop")

    for out in outputs:
        ttnn.deallocate(out)
