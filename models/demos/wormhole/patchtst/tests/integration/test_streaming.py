# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.demo.runner import run_streaming_forecast
from models.demos.wormhole.patchtst.tests.helpers import compute_metrics

PARITY_CORRELATION = 0.999
PARITY_MSE = 1e-2


@pytest.mark.timeout(600)
def test_patchtst_streaming_forecast_matches_direct_forward():
    cfg = merge_demo_config(
        PatchTSTDemoConfig(task="forecast"),
        task="forecast",
        strict_fallback=True,
        dataset="etth1",
        batch_size=1,
        max_windows=8,
        use_trace=True,
    )
    cached = run_streaming_forecast(config=cfg, stream_steps=4, streaming_mode="cached")
    full_rerun = run_streaming_forecast(config=cfg, stream_steps=4, streaming_mode="full-rerun")
    assert len(cached) == 4
    assert len(full_rerun) == 4
    for cached_step, full_step in zip(cached, full_rerun, strict=True):
        parity = compute_metrics(cached_step, full_step)
        assert parity["correlation"] >= PARITY_CORRELATION
        assert parity["mse"] <= PARITY_MSE


@pytest.mark.timeout(600)
def test_patchtst_streaming_forecast_longer_horizon_stability():
    cfg = merge_demo_config(
        PatchTSTDemoConfig(task="forecast"),
        task="forecast",
        strict_fallback=True,
        dataset="etth1",
        batch_size=1,
        max_windows=16,
        use_trace=True,
    )
    cached = run_streaming_forecast(config=cfg, stream_steps=12, streaming_mode="cached")
    full_rerun = run_streaming_forecast(config=cfg, stream_steps=12, streaming_mode="full-rerun")
    assert len(cached) == 12
    assert len(full_rerun) == 12
    for cached_step, full_step in zip(cached, full_rerun, strict=True):
        parity = compute_metrics(cached_step, full_step)
        assert parity["correlation"] >= PARITY_CORRELATION
        assert parity["mse"] <= PARITY_MSE
