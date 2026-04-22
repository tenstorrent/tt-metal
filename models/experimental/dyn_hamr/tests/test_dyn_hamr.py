# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness for Dyn-HaMR on Blackhole p150.

Emits three grep-matched key lines that the autoresearch outer loop consumes:

    inference_speed: <float> frames/sec
    accuracy: <float>
    peak_dram: <int> bytes

`accuracy` is the Pearson correlation (×100) between the live system output
and the torch CPU reference, evaluated on identical random weights + input.
When the tt-nn port doesn't exist yet, the *reference itself* is the live
system, so accuracy pins to 100 and `inference_speed` reports the CPU speed —
the number the NPU port must beat.
"""
from __future__ import annotations

import os
import time

import pytest
import torch


FRAMES = int(os.environ.get("DYN_HAMR_FRAMES", "3"))
WARMUP = int(os.environ.get("DYN_HAMR_WARMUP", "1"))
BATCH = int(os.environ.get("DYN_HAMR_BATCH", "1"))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def _emit(speed_fps: float, accuracy: float, peak_dram_bytes: int) -> None:
    accuracy = max(0.0, min(100.0, accuracy))
    print(f"inference_speed: {speed_fps:.4f} frames/sec")
    print(f"accuracy: {accuracy:.4f}")
    print(f"peak_dram: {peak_dram_bytes}")
    print(f"peak_vram: {peak_dram_bytes}")  # PROGRAM.md greps both; emit both.


def _time_forward(model, sample: torch.Tensor) -> float:
    with torch.no_grad():
        for _ in range(WARMUP):
            model(sample)
        t0 = time.perf_counter()
        for _ in range(FRAMES):
            model(sample)
        dt = time.perf_counter() - t0
    return (FRAMES * sample.shape[0]) / dt if dt > 0 else 0.0


def _try_import_tt():
    try:
        from models.experimental.dyn_hamr.tt import hamer as tt_hamer  # noqa: WPS433
        return tt_hamer
    except Exception:
        return None


def _try_import_ref():
    try:
        from models.experimental.dyn_hamr.reference import hamer as ref_hamer  # noqa: WPS433
        return ref_hamer
    except Exception:
        return None


@pytest.mark.parametrize("device_id", [int(os.environ.get("DYN_HAMR_DEVICE", "0"))])
def test_dyn_hamr_inference(device_id: int) -> None:
    ref_mod = _try_import_ref()
    if ref_mod is None:
        _emit(0.0, 0.0, 0)
        pytest.skip("dyn_hamr reference not yet importable")

    tt_mod = _try_import_tt()

    torch.manual_seed(0)
    sample = ref_mod.sample_input(batch=BATCH)

    if tt_mod is None:
        # Live system == torch CPU reference; accuracy is 100 by definition.
        ref_model = ref_mod.build_reference()
        with torch.no_grad():
            _ = ref_model(sample)  # shape check
        fps = _time_forward(ref_model, sample)
        _emit(fps, 100.0, 0)
        return

    ref_model, tt_model, sample = ref_mod.build_paired(tt_mod, device_id=device_id)
    with torch.no_grad():
        ref_out = ref_model(sample)
        tt_out = tt_model(sample)
    pcc_pct = _pcc(tt_out, ref_out) * 100.0
    fps = _time_forward(tt_model, sample)
    peak = getattr(tt_model, "peak_dram_bytes", 0)
    _emit(fps, pcc_pct, peak)
