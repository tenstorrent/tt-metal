#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""ResNet-50 ImageNet inference benchmark on Tenstorrent Wormhole via TTNN.

Native batch=1 support (branch: akannan/resnet_50_support_bs_1).

Run with pytest:
    pytest run_inference_modified.py -v -s
    pytest run_inference_modified.py -v -s -k single
    pytest run_inference_modified.py -v -s -k benchmark
    N_IMAGES=50 WARMUP=10 pytest run_inference_modified.py -v -s -k benchmark
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make repo root importable regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import ttnn

from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
    create_test_infra,
)
from helpers import run_logging

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

_LABELS_PATH = _REPO_ROOT / "models" / "sample_data" / "imagenet_class_labels.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    labels = _imagenet_labels()

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

    predictions = top5(logits_row0, labels)
    assert len(predictions) == 5, "Expected 5 top predictions"

    print(f"[Result] Top-5 predictions  (latency: {latency_ms:.1f} ms)")
    for rank, (label, prob) in enumerate(predictions, 1):
        print(f"  {rank}. {label:40s}  {prob * 100:.2f}%")
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_benchmark(device, test_infra):
    """Timed benchmark loop at native batch_size=1.

    Configure via environment variables:
        N_IMAGES  — number of timed batches  (default: 10)
        WARMUP    — warmup batches           (default: 5)
    """
    n_images = int(os.environ.get("N_IMAGES", "10"))
    warmup = int(os.environ.get("WARMUP", "5"))

    bs = _BATCH_SIZE
    script_tag = f"resnet50_ttnn_b{bs}"

    with run_logging(script_tag):
        pool_size = min(n_images + warmup, 8)
        _frames = [random_input(batch_size=bs) for _ in range(pool_size)]
        print(f"\n[benchmark] Input pool: {pool_size} tensors (shape {[bs, *INPUT_SHAPE_CHW]}, bfloat16)")

        from helpers import print_results_table

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
