# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""VGGT end-to-end benchmark and PCC accuracy test on Tenstorrent Blackhole.

Compares the ttnn port against the PyTorch reference model output via
Pearson Correlation Coefficient (PCC) on synthetic random inputs.

Environment variables:
    TT_METAL_HOME   — root of the tt-metal checkout (auto-detected if unset)
    TT_DEVICE_ID    — chip index, default 0
    VGGT_REF_PATH   — path to facebook/VGGT upstream source tree
    VGGT_WEIGHTS_PATH — path to model.safetensors from facebook/VGGT-1B
    VGGT_S_CANON    — canonical S for BF0 padding (default: matches --seq)

Usage:
    # PCC test at S=1
    pytest models/experimental/vggt/tests/test_vggt.py -v

    # Perf only
    pytest models/experimental/vggt/tests/test_vggt.py::test_perf_s1 -v -s

    # Direct (S=2)
    python models/experimental/vggt/tests/test_vggt.py --seq 2
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import pytest
import torch

# Ensure tt-metal root is importable (5 dirs up from this file).
_TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", str(Path(__file__).parents[4])))
if str(_TT_METAL_HOME) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_HOME))

from models.experimental.vggt.reference.torch_vggt import load_vggt  # noqa: E402
from models.experimental.vggt.tt.ttnn_vggt import (  # noqa: E402
    _ensure_installed,
    vggt_forward,
)

PCC_THRESHOLD = 0.99
_PCC_KEYS = ("depth", "depth_conf", "world_points", "world_points_conf", "pose_enc")
DEVICE_ID = int(os.environ.get("TT_DEVICE_ID", "0"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom else 0.0


def _multi_pcc(ref: dict, out: dict) -> tuple[float, dict[str, float]]:
    scores = {
        k: _pcc(ref[k], out[k])
        for k in _PCC_KEYS
        if k in ref and k in out
        and isinstance(ref[k], torch.Tensor)
    }
    return (min(scores.values()) if scores else 0.0), scores


def _device_peak_dram_mb(device) -> float:
    try:
        import ttnn._ttnn as _t  # type: ignore
        stats = _t.device.allocator_statistics(device, _t.tensor.BufferType.DRAM)
        return float(stats.peak_bytes) / (1024 * 1024)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tt_device():
    import ttnn
    dev = ttnn.open_device(device_id=DEVICE_ID, l1_small_size=32 * 1024)
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    _ensure_installed(dev, prewarm_seqs=(1, 2))
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def ref_model():
    model = load_vggt(eval_mode=True)
    yield model
    del model


# ---------------------------------------------------------------------------
# PCC tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", [1, 2])
def test_pcc_end_to_end(tt_device, ref_model, seq):
    """ttnn output must match torch reference at PCC >= 0.99 for all outputs."""
    torch.manual_seed(0)
    images = torch.rand(1, seq, 3, 518, 518)

    with torch.no_grad():
        ref_out = ref_model(images)

    tt_out = vggt_forward(images, device=tt_device)

    min_pcc, scores = _multi_pcc(ref_out, tt_out)
    print(f"\nS={seq} PCC: {scores}")
    assert min_pcc >= PCC_THRESHOLD, (
        f"S={seq} min PCC {min_pcc:.4f} < {PCC_THRESHOLD} — failing key: "
        + str({k: f"{v:.4f}" for k, v in scores.items() if v < PCC_THRESHOLD})
    )


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------

def test_perf_s1(tt_device):
    """Latency at B=1 S=1 must stay below 5000 ms (CPU torch baseline)."""
    torch.manual_seed(0)
    images = torch.rand(1, 1, 3, 518, 518)

    vggt_forward(images, device=tt_device)  # warmup

    times = [
        (lambda t0=time.perf_counter(): (vggt_forward(images, device=tt_device),
                                         (time.perf_counter() - t0) * 1000)[1])()
        for _ in range(3)
    ]
    latency_ms = min(times)
    fps = 1000.0 / latency_ms
    peak = _device_peak_dram_mb(tt_device)
    print(f"\nS=1 latency: {latency_ms:.1f} ms  |  {fps:.3f} fps  |  peak DRAM {peak:.0f} MB")
    print(f"CPU torch baseline: 5037 ms (0.199 fps)")
    assert latency_ms < 5000, f"Regression: {latency_ms:.1f} ms exceeds 5000 ms CPU baseline"


# ---------------------------------------------------------------------------
# Standalone CLI (mirrors the original test_vggt.py interface)
# ---------------------------------------------------------------------------

def _run_cli():
    parser = argparse.ArgumentParser(description="VGGT ttnn benchmark")
    parser.add_argument("--layer", default="end_to_end")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--prewarm-seqs", default="")
    args = parser.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()

    _closed = [False]
    def _close_once():
        if _closed[0]:
            return
        _closed[0] = True
        try:
            ttnn.close_device(device)
        except Exception:
            traceback.print_exc()

    def _sig(signum, _frame):
        print(f"\n[test_vggt] signal {signum}, closing device...", flush=True)
        _close_once()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        prewarm = (
            tuple(int(s) for s in args.prewarm_seqs.split(",") if s)
            if args.prewarm_seqs
            else ((args.seq,) if args.seq <= 2 else (1, 2))
        )
        _ensure_installed(device, prewarm_seqs=prewarm)

        torch.manual_seed(0)
        images = torch.rand(args.batch, args.seq, 3, args.img_size, args.img_size)
        ref_model = load_vggt(eval_mode=True)
        with torch.no_grad():
            ref_out = ref_model(images)
        del ref_model

        vggt_forward(images, device=device)  # warmup
        times = []
        tt_out = None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            tt_out = vggt_forward(images, device=device)
            times.append((time.perf_counter() - t0) * 1000)

        latency_ms = min(times)
        min_pcc, scores = _multi_pcc(ref_out, tt_out)
        peak = _device_peak_dram_mb(device)
        fps = args.seq * 1000.0 / latency_ms
        status = "PASS" if min_pcc >= PCC_THRESHOLD else "FAIL"

        print(f"--- layer: {args.layer}")
        print(f"pcc: {min_pcc:.4f}")
        for k, v in scores.items():
            print(f"pcc_{k}: {v:.4f}")
        print(f"latency_ms: {latency_ms:.2f}")
        print(f"inference_speed: {fps:.4f}")
        print(f"accuracy: {min(100.0, min_pcc * 100):.4f}")
        print(f"peak_dram: {peak:.2f}")
        print(f"status: {status}")
        print("---")
        return 0 if status == "PASS" else 1
    except Exception:
        traceback.print_exc()
        return 3
    finally:
        _close_once()


if __name__ == "__main__":
    sys.exit(_run_cli())
