# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import contextlib
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Union

import pytest
import torch

import ttnn
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import create_test_infra

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SHAPE_CHW = (3, 224, 224)

_BATCH_SIZE = 1

_MODEL_CONFIG = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Tee:
    """Proxy that writes to *original* and *log_file* simultaneously."""

    def __init__(self, original: IO[str], log_file: IO[str]) -> None:
        self._orig = original
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._orig.write(data)
        self._log.write(data)
        return n

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


@contextlib.contextmanager
def run_logging(script_name: str, log_dir: Union[str, Path] = "logs"):
    """Context manager: tee stdout + stderr to a timestamped log file."""
    log_dir_path = Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir_path / f"{timestamp}_{script_name}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _Tee(orig_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(orig_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"[logging] Run log: {log_path}")
            yield log_path
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.flush()

    print(f"[logging] Log saved → {log_path}")


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:8.2f} ± {std:6.2f} ms"


def print_results_table(
    n_timed: int,
    n_warmup: int,
    batch_size: int,
    prep_ms: list[float],
    infer_ms: list[float],
    collect_ms: list[float],
) -> tuple[float, float]:
    """Print the canonical benchmark results table and return ``(mean_ms, fps_per_image)``.

    Args:
        n_timed:    Number of timed frames.
        n_warmup:   Number of warmup frames (shown in header only).
        batch_size: Number of images per frame; used to compute per-image FPS.
        prep_ms:    Per-frame H2D latency list (ms).
        infer_ms:   Per-frame inference latency list (ms).
        collect_ms: Per-frame D2H latency list (ms).

    Returns:
        ``(mean_frame_ms, fps_per_image)``
    """
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_ms = statistics.mean(total_ms)
    fps_batch = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    fps_image = fps_batch * batch_size

    print()
    print("=" * 62)
    print(f"  Benchmark results  ({n_timed} frames, {n_warmup} warmup, batch={batch_size})")
    print("=" * 62)
    print(f"  H2D  (from_torch + to device)     :  {_fmt(prep_ms)}")
    print(f"  Run  (dispatch + synchronize)      :  {_fmt(infer_ms)}")
    print(f"  D2H  (to_torch)                    :  {_fmt(collect_ms)}")
    print("-" * 62)
    print(f"  Total per frame                    :  {_fmt(total_ms)}")
    print(f"  Frames/s                           :  {fps_batch:8.2f} fps")
    print(f"  Images/s  (× batch {batch_size:<3d})           :  {fps_image:8.2f} fps")
    print("=" * 62)

    return avg_ms, fps_image


def _imagenet_labels() -> dict[int, str]:
    with open(_LABELS_PATH) as f:
        return ast.literal_eval(f.read())


def random_input(batch_size: int = _BATCH_SIZE) -> torch.Tensor:
    """Return a Gaussian noise bfloat16 tensor of shape ``(B, 3, 224, 224)``."""
    return torch.randn(batch_size, *INPUT_SHAPE_CHW, dtype=torch.bfloat16)


def top5(logits: torch.Tensor, labels: dict[int, str]) -> list[tuple[str, float]]:
    """Convert 1-D or 2-D logits to top-5 ``(label, probability)`` pairs for row 0."""
    vec = logits.reshape(-1)[:1000].float()
    probs = torch.softmax(vec, dim=0)
    values, idxs = torch.topk(probs, 5)
    return [(labels[int(i)], float(v)) for i, v in zip(idxs, values)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    """Open and close a single-chip MeshDevice for the entire test module."""
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        l1_small_size=24576,
    )
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def test_infra(device):
    """Load ResNet-50 at batch_size=1 once for all tests in the module."""
    return create_test_infra(
        device=device,
        batch_size=_BATCH_SIZE,
        act_dtype=_MODEL_CONFIG["ACTIVATIONS_DTYPE"],
        weight_dtype=_MODEL_CONFIG["WEIGHTS_DTYPE"],
        math_fidelity=_MODEL_CONFIG["MATH_FIDELITY"],
        use_pretrained_weight=True,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_inference(device, test_infra):
    """Load model at batch=1, run one frame, verify output shape and top-5."""

    frame = random_input(batch_size=_BATCH_SIZE)
    print(f"\n[inference] Running single forward pass (batch={_BATCH_SIZE}) …")

    tt_input_host, input_mem_config = test_infra.setup_l1_sharded_input(device, frame)
    test_infra.input_tensor = tt_input_host.to(device, input_mem_config)

    t0 = time.perf_counter()
    tt_out = test_infra.run()
    ttnn.synchronize_device(device)
    latency_ms = (time.perf_counter() - t0) * 1e3

    logits = ttnn.to_torch(tt_out).float()
    logits_row0 = logits.reshape(_BATCH_SIZE, 1000)[0]

    assert logits_row0.shape == (1000,), f"Expected 1000 logits, got {logits_row0.shape}"


def test_benchmark(device, test_infra):
    """Timed benchmark loop at native batch_size=1.

    Configure via environment variables:
        N_IMAGES  — number of timed batches  (default: 10)
        WARMUP    — warmup batches           (default: 5)
    """
    n_images = 10
    warmup = 5

    bs = _BATCH_SIZE
    script_tag = f"resnet50_ttnn_b{bs}"

    with run_logging(script_tag):
        pool_size = min(n_images + warmup, 8)
        _frames = [random_input(batch_size=bs) for _ in range(pool_size)]
        print(f"\n[benchmark] Input pool: {pool_size} tensors (shape {[bs, *INPUT_SHAPE_CHW]}, bfloat16)")

        prep_ms, infer_ms, collect_ms = [], [], []

        def _h2d(frame):
            tt_host, mem_cfg = test_infra.setup_l1_sharded_input(device, frame)
            return tt_host.to(device, mem_cfg)

        def _run_device(tt_on_dev):
            test_infra.input_tensor = tt_on_dev
            out = test_infra.run()
            ttnn.synchronize_device(device)
            return out

        def _d2h(tt_out):
            return ttnn.to_torch(tt_out).float().reshape(bs, 1000)

        if warmup > 0:
            print(f"[benchmark] Warmup ({warmup} batches) …")
            for i in range(warmup):
                ti = _h2d(_frames[i % len(_frames)])
                to = _run_device(ti)
                _d2h(to)
            print("[benchmark] Warmup done.")

        print(f"[benchmark] Timed run ({n_images} batches × {bs} images) …")
        for i in range(n_images):
            f = _frames[(warmup + i) % len(_frames)]

            t0 = time.perf_counter()
            ti = _h2d(f)
            prep_ms.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            to = _run_device(ti)
            infer_ms.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _d2h(to)
            collect_ms.append((time.perf_counter() - t0) * 1e3)

            pct = int((i + 1) / n_images * 100)
            print(f"\r[benchmark] {i + 1:4d}/{n_images}  ({pct:3d}%)", end="", flush=True)
        print()

        print_results_table(n_images, warmup, bs, prep_ms, infer_ms, collect_ms)

    assert len(infer_ms) == n_images, f"Expected {n_images} measurements, got {len(infer_ms)}"
    assert all(t > 0 for t in infer_ms), "All inference times must be positive"
