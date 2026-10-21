# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks: TTNN Timestral-209M against the upstream CPU reference on one chip.

Shapes (one decode block = one ``predict_block`` call):
    paper         1024 univariate series, context 2048, horizon 64 (the Chronos-2 paper shape)
    long_context  32 univariate series, context 32768 (the model maximum), horizon 1024 (a full block)
    multivariate  128 series x 8 variates, context 2048, horizon 64 (blocked variate attention)

``test_forecast_perf`` times the user-facing ``forecast`` call, which adds symmetric averaging
(two passes) and block-autoregressive decode on top of ``predict_block``.
Timings are wall clock with the program cache warm; the CPU reference uses every host thread.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import pytest
import torch

from models.experimental.timestral.tests.golden_helpers import load_checkpoint, series

STEADY_ITERS = 3
_CPU_CACHE: dict[str, tuple[float, torch.Tensor]] = {}


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    variates: int
    context: int
    horizon_patches: int

    @property
    def rows(self) -> int:
        return self.batch * self.variates


SHAPES = {
    s.name: s
    for s in (
        Shape("paper", 1024, 1, 2048, 2),
        Shape("long_context", 32, 1, 32768, 32),
        Shape("multivariate", 128, 8, 2048, 2),
    )
}


def _inputs(shape: Shape):
    context = series(shape.batch, shape.variates, shape.context).reshape(shape.rows, -1)
    observed = torch.ones_like(context, dtype=torch.bool)
    group_ids = torch.arange(shape.batch).repeat_interleave(shape.variates)
    return context, observed, group_ids


def _sync_time(device, fn):
    import ttnn

    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - start


def _cpu_reference(model, shape: Shape) -> tuple[float, torch.Tensor]:
    """One CPU ``predict_block``, cached per shape so both precisions share it."""
    if shape.name not in _CPU_CACHE:
        context, observed, group_ids = _inputs(shape)
        with torch.no_grad():
            start = time.perf_counter()
            out = model.predict_block(context, observed, group_ids, shape.horizon_patches)
            _CPU_CACHE[shape.name] = (time.perf_counter() - start, out)
    return _CPU_CACHE[shape.name]


def _pcc(expected: torch.Tensor, got: torch.Tensor) -> float:
    from tests.ttnn.utils_for_testing import comp_pcc

    return comp_pcc(expected.float(), got.float(), 0.0)[1]


def _tt_model(mesh_device, model, precision_name):
    from models.experimental.timestral.tt.common import TtTimestralPrecision
    from models.experimental.timestral.tt.model import TtTimestral

    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip model")
    precision = TtTimestralPrecision.performance() if precision_name == "performance" else TtTimestralPrecision()
    return TtTimestral.from_reference(mesh_device, model, precision)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("precision_name", ["accuracy", "performance"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_predict_block_perf(mesh_device, shape_name, precision_name):
    import ttnn

    from models.experimental.timestral.tt.common import pad_to_tile

    shape = SHAPES[shape_name]
    model = load_checkpoint("Timestral-209M")
    tt = _tt_model(mesh_device, model, precision_name)
    context, observed, group_ids = _inputs(shape)
    h = shape.horizon_patches

    def run():
        return tt.predict_block(context, observed, group_ids, h)

    _, cold_s = _sync_time(mesh_device, run)
    steady = []
    for _ in range(STEADY_ITERS):
        got, seconds = _sync_time(mesh_device, run)
        steady.append(seconds)
    steady_s = statistics.median(steady)

    start = time.perf_counter()
    prepared = tt.prepare_inputs(context, observed, group_ids, h)
    prepare_s = time.perf_counter() - start
    device_inputs, upload_s = _sync_time(mesh_device, lambda: tt.upload_inputs(prepared))
    out, device_s = _sync_time(mesh_device, lambda: tt.forward_device(device_inputs))
    start = time.perf_counter()
    tt.postprocess(out, prepared)
    postprocess_s = time.perf_counter() - start
    ttnn.deallocate(out)
    tt.deallocate_inputs(device_inputs)

    cpu_s, expected = _cpu_reference(model, shape)
    normalized = lambda q: torch.asinh((q - prepared.anchor_loc[..., None]) / prepared.anchor_scale[..., None])
    pcc_norm = _pcc(normalized(expected), normalized(got))
    pcc_raw = _pcc(expected, got)

    padded_seq = pad_to_tile(prepared.seq_len)
    print(
        f"\n[PERF] predict_block {shape.name} / {precision_name}"
        f"\n  rows x context -> horizon: {shape.rows} x {shape.context} -> {h * model.config.patch_size}"
        f" ({shape.batch} series x {shape.variates} variates)"
        f"\n  sequence:     {prepared.seq_len} tokens per row, {padded_seq} padded"
        f"\n  cold_s:       {cold_s:.3f}"
        f"\n  steady_s:     {steady_s:.3f}  (median of {STEADY_ITERS}: {', '.join(f'{s:.3f}' for s in steady)})"
        f"\n  series_per_s: {shape.batch / steady_s:.1f}"
        f"\n  cpu_s:        {cpu_s:.3f}  ({torch.get_num_threads()} threads)"
        f"\n  speedup:      {cpu_s / steady_s:.1f}x vs CPU"
        f"\n  pcc:          {pcc_norm:.5f} normalized, {pcc_raw:.5f} raw"
        f"\n  split:        prepare {prepare_s:.3f}  upload {upload_s:.3f}  device {device_s:.3f}"
        f"  postprocess {postprocess_s:.3f}"
    )
    assert pcc_norm >= 0.98, f"{shape.name}/{precision_name}: normalized PCC {pcc_norm}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch,context_len,prediction_length", [(1024, 2048, 64), (32, 4096, 2048)], ids=["paper", "long_horizon"]
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_forecast_perf(mesh_device, batch, context_len, prediction_length):
    """End to end ``forecast``: symmetric averaging plus ceil(prediction_length / 1024) decode blocks."""
    from timestral.forecaster import TimestralForecaster

    from models.experimental.timestral.tt.model import TtTimestralForecaster

    model = load_checkpoint("Timestral-209M")
    tt = _tt_model(mesh_device, model, "accuracy")
    context = series(batch, 1, context_len, seed=3)
    tt_forecaster = TtTimestralForecaster(tt)

    _, cold_s = _sync_time(mesh_device, lambda: tt_forecaster.forecast(context, prediction_length=prediction_length))
    got, steady_s = _sync_time(
        mesh_device, lambda: tt_forecaster.forecast(context, prediction_length=prediction_length)
    )
    start = time.perf_counter()
    expected = TimestralForecaster(model).forecast(context, prediction_length=prediction_length)
    cpu_s = time.perf_counter() - start

    pcc = _pcc(expected.median, got.median)
    print(
        f"\n[PERF] forecast {batch} x {context_len} -> {prediction_length}"
        f"\n  cold_s:       {cold_s:.3f}"
        f"\n  steady_s:     {steady_s:.3f}"
        f"\n  series_per_s: {batch / steady_s:.1f}"
        f"\n  cpu_s:        {cpu_s:.3f}  ({torch.get_num_threads()} threads)"
        f"\n  speedup:      {cpu_s / steady_s:.1f}x vs CPU"
        f"\n  median pcc:   {pcc:.5f}"
    )
    assert pcc >= 0.98, f"forecast median PCC {pcc}"
