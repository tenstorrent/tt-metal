#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""ResNet-50 ImageNet inference benchmark on Tenstorrent Wormhole via TTNN.

Mirrors the Forge/ONNX benchmark at:
  acs-sw-arch/acs-demo/ai-chiplet-migration/tenstorrent/resnet50/run_inference.py

Model:   torchvision ResNet-50  (IMAGENET1K_V1 pretrained weights)
Input:   B × 3 × 224 × 224  float32 NCHW  (Gaussian noise for benchmarking)
Output:  B × 1000  logits  (ImageNet-1k)

── Hardware findings ─────────────────────────────────────────────────────────

Wormhole B0 sharding constraint (why batch=1 does not run natively):

  1. ttnn_functional_resnet50.__init__ only configures fold/shard grids for
     batch 16, 20, and 32.  batch_size=1 is never assigned fold_compute_grid_size
     and ResNet50TestInfra.setup_l1_sharded_input() has no branch for it.
     The test infrastructure explicitly calls:
       pytest.skip("Batch size 1 and 2 are not supported with sharded data")

  2. The run() method on Wormhole B0 applies three L1-sharding remaps whose
     shard heights are multiples of (spatial × batch) that only divide evenly
     by the tile size (32 rows) for batch ≥ 16:

       After max_pool2d  : CoreGrid(x=8, y=7) → 56 cores, 56×56×1 / 56 =  56 rows/shard ✗  (56 % 32 ≠ 0)
       After layer2      : CoreGrid(x=8, y=8) → 64 cores, 28×28×1 / 64 ≈ 12 rows/shard ✗
       After layer3      : CoreGrid(x=8, y=7) → 56 cores, 14×14×1 / 56 ≈  4 rows/shard ✗

     For batch=16 all three produce tile-aligned shards
     (e.g. 56×56×16 / 56 = 896 rows, 896 % 32 = 0 ✓).

  3. The bottleneck conv2 in layers 1 and 2 is called with height_sharding=True
     and enable_activation_reuse=True (hard-coded for stride==1).  This requires
     act_block_h_ntiles > output_image_width_ntiles, which fails at batch=1
     where both values collapse to 2 tiles.

Fix applied here — batch=1 pads to batch=16:
  pad_to_batch() tiles the single image 16× and runs the full sharded pipeline.
  Per-image metrics are reported as batch16_value / 16.

── Usage ─────────────────────────────────────────────────────────────────────

    python run_inference.py                                     # single image, top-5
    python run_inference.py --benchmark --n-images 10 --warmup 5
    python run_inference.py --benchmark --batch-size 16         # explicit b16
    python run_inference.py --benchmark --n-images 50 --warmup 10
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

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
INPUT_SHAPE_CHW = (3, 224, 224)  # per-image; batch dimension added at runtime

# Minimum hardware-supported batch for Wormhole B0 sharded execution.
_HW_BATCH = 16

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


def random_input(batch_size: int = _HW_BATCH) -> torch.Tensor:
    """Return a Gaussian noise float32 tensor of shape ``(B, 3, 224, 224)``."""
    return torch.randn(batch_size, *INPUT_SHAPE_CHW, dtype=torch.bfloat16)


def pad_to_batch(frame: torch.Tensor, batch_size: int = _HW_BATCH) -> torch.Tensor:
    """Tile a single-image (1, C, H, W) tensor to (batch_size, C, H, W)."""
    assert frame.shape[0] == 1, "pad_to_batch expects a 1-image tensor"
    return frame.expand(batch_size, -1, -1, -1).contiguous()


def top5(logits: torch.Tensor, labels: dict[int, str]) -> list[tuple[str, float]]:
    """Convert 1-D or 2-D logits to top-5 ``(label, probability)`` pairs for row 0."""
    vec = logits.reshape(-1)[:1000].float()
    probs = torch.softmax(vec, dim=0)
    values, idxs = torch.topk(probs, 5)
    return [(labels[int(i)], float(v)) for i, v in zip(idxs, values)]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def open_device() -> ttnn.MeshDevice:
    """Open a single-chip MeshDevice with the memory config required by ResNet50."""
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        l1_small_size=24576,
    )


# ---------------------------------------------------------------------------
# Model loading (always batch=16 or explicit supported batch)
# ---------------------------------------------------------------------------


def _load_model(device: ttnn.MeshDevice, batch_size: int = _HW_BATCH):
    """Load ResNet-50 via ResNet50TestInfra for the given batch size."""
    return create_test_infra(
        device=device,
        batch_size=batch_size,
        act_dtype=_MODEL_CONFIG["ACTIVATIONS_DTYPE"],
        weight_dtype=_MODEL_CONFIG["WEIGHTS_DTYPE"],
        math_fidelity=_MODEL_CONFIG["MATH_FIDELITY"],
        use_pretrained_weight=True,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=None,
    )


# ---------------------------------------------------------------------------
# Single-shot inference
# ---------------------------------------------------------------------------


def _run_single_inference(device: ttnn.MeshDevice) -> None:
    """Load model, run one frame (padded to batch=16), print top-5 predictions.

    Per-image latency is reported as ``batch16_latency / 16``.
    """
    print(f"[model] Loading ResNet-50 weights (batch={_HW_BATCH}) …")
    test_infra = _load_model(device, _HW_BATCH)
    labels = _imagenet_labels()

    # Single frame padded to the hardware batch size
    frame1 = random_input(batch_size=1)
    batch = pad_to_batch(frame1, _HW_BATCH)
    print(f"[inference] Running single forward pass (batch={_HW_BATCH}, reporting image [0]) …")

    tt_input_host, input_mem_config = test_infra.setup_l1_sharded_input(device, batch)
    test_infra.input_tensor = tt_input_host.to(device, input_mem_config)

    t0 = time.perf_counter()
    tt_out = test_infra.run()
    ttnn.synchronize_device(device)
    batch_ms = (time.perf_counter() - t0) * 1e3
    per_image_ms = batch_ms / _HW_BATCH

    logits_batch = ttnn.to_torch(tt_out).float()  # (16, 1, 1, 1000) or similar
    logits_row0 = logits_batch.reshape(_HW_BATCH, 1000)[0]

    print(f"\n[Result] Top-5 predictions  (per-image latency: {per_image_ms:.1f} ms, batch latency: {batch_ms:.1f} ms)")
    for rank, (label, prob) in enumerate(top5(logits_row0, labels), 1):
        print(f"  {rank}. {label:40s}  {prob * 100:.2f}%")


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------


def _run_benchmark(args: argparse.Namespace, device: ttnn.MeshDevice) -> None:
    """Timed benchmark loop.

    For ``--batch-size 1``:  internally runs batch=16, reports per-image metrics.
    For ``--batch-size 16/20/32``:  runs at that batch size directly.
    """
    requested_bs = args.batch_size
    hw_bs = max(requested_bs, _HW_BATCH)  # enforce minimum hardware batch

    if requested_bs != hw_bs:
        print(
            f"[info] --batch-size {requested_bs} → padded to hardware batch {hw_bs} "
            f"(per-image metrics reported as value / {hw_bs})"
        )

    script_tag = f"resnet50_ttnn_b{hw_bs}" if requested_bs == hw_bs else f"resnet50_ttnn_padded{hw_bs}"

    with run_logging(script_tag):
        print(f"[model] Loading ResNet-50 weights (batch={hw_bs}) …")
        test_infra = _load_model(device, hw_bs)

        pool_size = min(args.n_images + args.warmup, 8)
        if requested_bs == 1:
            # Single images padded to hw_bs
            _frames = [pad_to_batch(random_input(batch_size=1), hw_bs) for _ in range(pool_size)]
        else:
            _frames = [random_input(batch_size=hw_bs) for _ in range(pool_size)]
        print(f"[benchmark] Input pool: {pool_size} tensors (shape {[hw_bs, *INPUT_SHAPE_CHW]}, float32)")

        import statistics as _stats
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
            return ttnn.to_torch(tt_out).float().reshape(hw_bs, 1000)

        if args.warmup > 0:
            print(f"[benchmark] Warmup ({args.warmup} batches) …")
            for i in range(args.warmup):
                ti = _h2d(_frames[i % len(_frames)])
                to = _run_device(ti)
                _d2h(to)
            print("[benchmark] Warmup done.")

        print(f"[benchmark] Timed run ({args.n_images} batches × {hw_bs} images) …")
        for i in range(args.n_images):
            f = _frames[(args.warmup + i) % len(_frames)]

            t0 = time.perf_counter()
            ti = _h2d(f)
            prep_ms.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            to = _run_device(ti)
            infer_ms.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _d2h(to)
            collect_ms.append((time.perf_counter() - t0) * 1e3)

            pct = int((i + 1) / args.n_images * 100)
            print(f"\r[benchmark] {i + 1:4d}/{args.n_images}  ({pct:3d}%)", end="", flush=True)
        print()

        print_results_table(args.n_images, args.warmup, hw_bs, prep_ms, infer_ms, collect_ms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 16, 20, 32],
        help=(
            "Logical batch size. 1 = pad to 16 internally, report per-image metrics. "
            "16/20/32 = run at that batch size. Default: 1"
        ),
    )

    bench = p.add_argument_group("benchmark")
    bench.add_argument(
        "--benchmark",
        action="store_true",
        help="Run timed benchmark loop instead of a single inference",
    )
    bench.add_argument(
        "--n-images",
        type=int,
        default=10,
        help="Number of timed batches / frames (default: 10)",
    )
    bench.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup batches, not included in timing (default: 5)",
    )

    args = p.parse_args()

    device = open_device()
    try:
        if args.benchmark:
            _run_benchmark(args, device)
        else:
            _run_single_inference(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
