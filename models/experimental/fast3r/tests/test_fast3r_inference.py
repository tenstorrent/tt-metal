"""Fast3R benchmark harness.

Prints `inference_speed`, `accuracy`, `peak_dram` keywords on stdout so the
autoresearch loop can `grep` for them after a run.

Initial iteration: encoder+decoder CPU reference only (no tt-nn), used to
establish a working baseline. Subsequent iterations will swap components over
to tt-nn one at a time.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.experimental.fast3r.reference.weights import load_fast3r


def _make_input(batch: int, img_size: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, img_size, img_size, generator=g)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((num / den).item())


def _peak_dram_bytes() -> int:
    # No ttnn device in this iteration; placeholder for future tt-nn runs.
    return 0


@pytest.mark.parametrize("n_views", [1])
@pytest.mark.parametrize("img_size", [512])
def test_fast3r_baseline(n_views: int, img_size: int):
    torch.set_num_threads(int(os.environ.get("FAST3R_TORCH_THREADS", "16")))
    model = load_fast3r(device="cpu")
    x = _make_input(n_views, img_size)

    # Warm-up
    with torch.inference_mode():
        ref_out = model(x)

    # Timed run — average over a few iters
    iters = int(os.environ.get("FAST3R_BENCH_ITERS", "3"))
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            out = model(x)
    dt = (time.perf_counter() - t0) / iters
    fps = n_views / dt if dt > 0 else 0.0

    # Self-PCC as a placeholder until the tt-nn port produces an output to compare against.
    acc = _pcc(out, ref_out) * 100.0

    print(f"inference_speed: {fps:.4f} views/sec")
    print(f"accuracy: {acc:.2f}")
    print(f"peak_dram: {_peak_dram_bytes()}")
