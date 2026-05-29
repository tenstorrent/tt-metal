# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Benchmark the qwen35_27b VisionTransformer in two configurations:

  1. ``replicated`` -- the default. Every device runs a full copy of the
     vision model with replicated weights (i.e. the current behavior).
  2. ``tensor_parallel`` -- weights are sharded across the mesh via the
     Megatron-style TP path added in `vision_attention_tp.py` /
     `vision_mlp_tp.py`.

Both configurations are timed end-to-end on a single image (full vision tower
prefill, including patch_merger). The runs share a mesh_device fixture so the
two paths are timed back-to-back; results are printed in a table at the end.

Run:
    MESH_DEVICE=T3K pytest -xvs \
        models/demos/qwen35_27b/tests/benchmark_vision_tp.py

Useful knobs (env vars):
    QWEN_VISION_BENCH_GRID    Image grid as "T,H,W" (default "1,86,128")
    QWEN_VISION_BENCH_LAYERS  Number of vision layers (default = full depth)
    QWEN_VISION_BENCH_WARMUP  Warmup iterations (default 2)
    QWEN_VISION_BENCH_ITERS   Timed iterations (default 5)
"""

import gc
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.vision.model import DropInVisionTransformer
from models.demos.qwen35_27b.tt.vision.vision_model_config import VisionModelArgs


def _bench_one_mode(
    *,
    mesh_device,
    reference_model,
    image_grid_thw: torch.Tensor,
    pt_pixel_values: torch.Tensor,
    vision_tp: bool,
    num_layers: int,
    warmup: int,
    iters: int,
    dtype,
):
    """Construct a DropInVisionTransformer in the requested mode and time it.

    Returns a dict with the recorded latencies (ms) and other metadata.
    """
    label = "tensor_parallel" if vision_tp else "replicated"
    logger.info(f"[bench] building {label} model (depth={num_layers}, vision_tp={vision_tp}) ...")

    # Compute the padded sequence length to mirror DropInVisionTransformer.
    grid = image_grid_thw[0]
    unpadded_seq_len = (grid[1] * grid[2]).item()
    seq_len = ((unpadded_seq_len // 2048) + 1) * 2048

    model_args = VisionModelArgs(
        mesh_device,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=seq_len,
        vision_tp=vision_tp,
    )
    if num_layers is not None:
        model_args.hf_config.vision_config.depth = num_layers

    tt_model = DropInVisionTransformer(reference_model, model_args, dtype=dtype)

    # Warmup -- excluded from timing. The first call also pays compile-time costs.
    for w in range(warmup):
        out = tt_model(pt_pixel_values, image_grid_thw)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
        logger.info(f"[bench/{label}] warmup {w + 1}/{warmup} done")

    # Timed runs.
    timings_ms = []
    for i in range(iters):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = tt_model(pt_pixel_values, image_grid_thw)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        ttnn.deallocate(out)
        ms = (t1 - t0) * 1000
        timings_ms.append(ms)
        logger.info(f"[bench/{label}] iter {i + 1}/{iters}: {ms:.2f} ms")

    # Drop the model so weights are freed before the next mode runs.
    del tt_model
    gc.collect()

    return {
        "label": label,
        "vision_tp": vision_tp,
        "num_layers": num_layers,
        "seq_len_padded": seq_len,
        "seq_len_unpadded": int(unpadded_seq_len),
        "timings_ms": timings_ms,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def _format_table(results):
    header = (
        f"{'mode':<18}{'depth':>7}{'seq_len':>10}{'min(ms)':>12}" f"{'median(ms)':>14}{'max(ms)':>12}{'samples':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        lines.append(
            f"{r['label']:<18}{r['num_layers']:>7}{r['seq_len_padded']:>10}"
            f"{r['min_ms']:>12.2f}{r['median_ms']:>14.2f}{r['max_ms']:>12.2f}{len(r['timings_ms']):>10}"
        )
    lines.append(sep)
    if len(results) == 2:
        rep = next(r for r in results if not r["vision_tp"])
        tp = next(r for r in results if r["vision_tp"])
        speedup = rep["median_ms"] / tp["median_ms"] if tp["median_ms"] > 0 else float("inf")
        lines.append(f"  TP speedup vs replicated (median): {speedup:.2f}x")
        lines.append(sep)
    return "\n".join(lines)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_benchmark_vision_tp_vs_replicated(mesh_device, reset_seeds, ensure_gc):
    """Run the vision tower in both modes back-to-back and print a comparison."""

    grid_str = os.environ.get("QWEN_VISION_BENCH_GRID", "1,128,86")
    image_grid_thw = torch.tensor([[int(x) for x in grid_str.split(",")]])
    assert image_grid_thw.shape == (1, 3), f"QWEN_VISION_BENCH_GRID must be 'T,H,W'; got {grid_str!r}"

    num_layers_env = os.environ.get("QWEN_VISION_BENCH_LAYERS")
    num_layers = int(num_layers_env) if num_layers_env else None

    warmup = int(os.environ.get("QWEN_VISION_BENCH_WARMUP", "2"))
    iters = int(os.environ.get("QWEN_VISION_BENCH_ITERS", "5"))

    dtype = ttnn.bfloat8_b

    # The reference HF vision tower is reused across both modes -- we only swap
    # the TT side. With dummy weights it is cheap enough to construct once.
    bootstrap_args = VisionModelArgs(
        mesh_device,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=2048,
    )
    target_depth = num_layers or bootstrap_args.hf_config.vision_config.depth
    if num_layers is not None:
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    reference_model = bootstrap_args.reference_vision_model(depth=target_depth)

    in_channels = bootstrap_args.hf_config.vision_config.in_channels
    patch_size = bootstrap_args.hf_config.vision_config.patch_size
    temporal_patch_size = bootstrap_args.hf_config.vision_config.temporal_patch_size
    pixel_dim = in_channels * patch_size * patch_size * temporal_patch_size
    n_patches = int((image_grid_thw[0, 0] * image_grid_thw[0, 1] * image_grid_thw[0, 2]).item())
    pt_pixel_values = torch.randn([n_patches, pixel_dim])

    logger.info(
        f"[bench] mesh={list(mesh_device.shape)} grid={image_grid_thw.tolist()} "
        f"depth={target_depth} warmup={warmup} iters={iters}"
    )

    results = []
    for vision_tp in (False, True):
        results.append(
            _bench_one_mode(
                mesh_device=mesh_device,
                reference_model=reference_model,
                image_grid_thw=image_grid_thw,
                pt_pixel_values=pt_pixel_values,
                vision_tp=vision_tp,
                num_layers=target_depth,
                warmup=warmup,
                iters=iters,
                dtype=dtype,
            )
        )

    table = _format_table(results)
    logger.info("\n" + table)
    print("\n" + table)
