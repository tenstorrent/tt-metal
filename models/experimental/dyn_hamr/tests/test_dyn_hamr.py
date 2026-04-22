# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness for Dyn-HaMR on Blackhole p150.

Emits three grep-matched key lines that the autoresearch outer loop consumes:

    inference_speed: <float> frames/sec
    accuracy: <float>
    peak_dram: <int> bytes

`accuracy` is the Pearson correlation (×100) between the tt-nn forward and the
torch CPU reference, evaluated on identical random weights + input.  A value
of 100 means exact match; the harness caps the reported accuracy at 100.

Iteration 0: no tt-nn port exists yet — the tt-nn branch is skipped, the
reference is exercised only to confirm the harness plumbing works, and the
reported speed is the CPU fallback (not the metric target).  Subsequent
iterations wire in the tt-nn impl and replace the zeros.
"""
from __future__ import annotations

import os
import time

import pytest
import torch


FRAMES = int(os.environ.get("DYN_HAMR_FRAMES", "4"))
WARMUP = int(os.environ.get("DYN_HAMR_WARMUP", "1"))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def _emit(speed_fps: float, accuracy: float, peak_dram_bytes: int) -> None:
    # Clamp accuracy to [0, 100] for a stable metric surface.
    accuracy = max(0.0, min(100.0, accuracy))
    print(f"inference_speed: {speed_fps:.4f} frames/sec")
    print(f"accuracy: {accuracy:.4f}")
    print(f"peak_dram: {peak_dram_bytes}")
    # PROGRAM.md alternately greps peak_vram; emit both for safety.
    print(f"peak_vram: {peak_dram_bytes}")


def _have_tt_impl() -> bool:
    try:
        from models.experimental.dyn_hamr.tt import hamer as _  # noqa: F401
        return True
    except Exception:
        return False


def _have_reference() -> bool:
    try:
        from models.experimental.dyn_hamr.reference import hamer as _  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.parametrize("device_id", [int(os.environ.get("DYN_HAMR_DEVICE", "0"))])
def test_dyn_hamr_inference(device_id: int) -> None:
    torch.manual_seed(0)

    if not _have_reference() or not _have_tt_impl():
        # Iteration 0 / early iterations: scaffolding only. Emit a well-formed
        # zero baseline so the autoresearch parser produces a clean row.
        _emit(0.0, 0.0, 0)
        pytest.skip("dyn_hamr reference or tt-nn impl not yet implemented")

    # Wired path (filled in by later iterations):
    from models.experimental.dyn_hamr.reference import hamer as ref_hamer
    from models.experimental.dyn_hamr.tt import hamer as tt_hamer

    ref_model, tt_model, sample = ref_hamer.build_paired(tt_hamer, device_id=device_id)

    # Accuracy via PCC on matched random input.
    with torch.no_grad():
        ref_out = ref_model(sample)
        tt_out = tt_model(sample)
    pcc = _pcc(tt_out, ref_out) * 100.0

    # Warmup then timed frames on tt-nn.
    with torch.no_grad():
        for _ in range(WARMUP):
            tt_model(sample)
        t0 = time.perf_counter()
        for _ in range(FRAMES):
            tt_model(sample)
        dt = time.perf_counter() - t0
    fps = FRAMES / dt if dt > 0 else 0.0

    peak = getattr(tt_model, "peak_dram_bytes", 0)
    _emit(fps, pcc, peak)
