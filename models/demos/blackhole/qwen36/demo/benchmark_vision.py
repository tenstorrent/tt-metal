# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone latency/throughput benchmark for the Qwen3.5-9B TT vision tower.

This drives ``DropInVisionTransformer`` (the drop-in replacement for the HF
``Qwen3_5_VisionTransformerPretrainedModel``) on synthetic pixel inputs and
reports per-image latency plus patch / image-token throughput. Only the vision
tower is exercised — no text model, no prefill/decode.

Examples
--------
Single device (default mesh (1,1)), full-depth, real weights::

    python models/demos/blackhole/qwen36/demo/benchmark_vision.py

Quick iteration with random weights and a couple of layers::

    python models/demos/blackhole/qwen36/demo/benchmark_vision.py \
        --dummy-weights --num-layers 2 --iters 5

A 4-device tensor-parallel mesh, custom image grids (t,h,w patches), more iters::

    python models/demos/blackhole/qwen36/demo/benchmark_vision.py \
        --mesh-shape 1 4 --grid 1,86,128 --grid 1,40,40 --iters 20

Notes
-----
* ``grid`` is the patch grid (temporal, height, width) BEFORE the 2x2 spatial
  merge. ``h`` and ``w`` must be divisible by ``spatial_merge_size`` (2).
* Compute performance is independent of weight *values*, so ``--dummy-weights``
  gives representative timings while skipping the large checkpoint download. The
  HF reference is still loaded (it owns the patch-embed / positional-embed steps
  that are not ported to TT).
"""

import argparse
import statistics
import time

import torch
from loguru import logger

import ttnn


def _parse_grid(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--grid expects 't,h,w', got '{s}'")
    return tuple(int(p) for p in parts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mesh-shape",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROWS", "COLS"),
        help="Mesh device shape. Cols>1 runs the vision tower tensor-parallel. Default: 1 1.",
    )
    p.add_argument(
        "--grid",
        type=_parse_grid,
        action="append",
        default=None,
        help="Patch grid 't,h,w' (pre-merge). Repeatable for multiple images. "
        "Default: a single 1,86,128 image (the demo image).",
    )
    p.add_argument(
        "--num-layers", type=int, default=None, help="Override vision depth (fewer = faster). Default: full."
    )
    p.add_argument("--iters", type=int, default=10, help="Timed iterations. Default: 10.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup (untimed) iterations for compile/cache. Default: 2.")
    p.add_argument(
        "--dtype",
        choices=["bfp8", "bf16"],
        default="bfp8",
        help="Vision weight/compute dtype. Default: bfp8 (bfloat8_b).",
    )
    p.add_argument(
        "--dummy-weights",
        action="store_true",
        help="Use random weights (skips the checkpoint download; perf is unchanged).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run the HF reference alongside and log PCC (slow; correctness sanity check).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rows, cols = args.mesh_shape
    multi = rows * cols > 1
    dtype = ttnn.bfloat8_b if args.dtype == "bfp8" else ttnn.bfloat16

    grids = args.grid if args.grid else [(1, 86, 128)]
    grid_thw = torch.tensor(grids, dtype=torch.long)

    # Padded sequence length the wrapper rounds each image up to (must divide 2048),
    # used to size the model config / activations.
    max_unpadded = max(t * h * w for (t, h, w) in grids)
    max_seq_len = ((max_unpadded // 2048) + 1) * 2048

    logger.info(f"Opening mesh device {rows}x{cols} ({'TP' if multi else 'single'})")
    if multi:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        l1_small_size=24576,
    )

    try:
        try:
            mesh_device.enable_program_cache()
        except AttributeError:
            pass

        # Imported here so the device is open / fabric is configured first.
        from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
        from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

        logger.info("Building VisionModelArgs...")
        model_args = VisionModelArgs(
            mesh_device,
            dummy_weights=args.dummy_weights,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        vc = model_args.hf_config.vision_config
        merge = vc.spatial_merge_size
        for t, h, w in grids:
            assert h % merge == 0 and w % merge == 0, f"grid h,w must be divisible by spatial_merge_size={merge}"

        depth = args.num_layers if args.num_layers is not None else vc.depth
        vc.depth = depth

        logger.info(f"Loading HF reference vision model (depth={depth})...")
        t0 = time.time()
        reference_model = model_args.reference_vision_model(depth=depth)
        logger.info(f"Reference model loaded in {time.time() - t0:.1f}s")

        logger.info(f"Building DropInVisionTransformer (dtype={args.dtype})...")
        t0 = time.time()
        model = DropInVisionTransformer(reference_model, model_args, dtype=dtype, debug=args.debug)
        logger.info(f"TT vision tower built in {time.time() - t0:.1f}s")

        # Synthetic patch inputs: [total_patches, in_channels * temporal_patch_size * patch_size**2].
        pixel_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        total_patches = int(grid_thw.prod(dim=-1).sum().item())
        total_image_tokens = total_patches // (merge**2)
        pixel_values = torch.randn(total_patches, pixel_dim)

        logger.info(
            f"Inputs: {len(grids)} image(s), grids={grids}, "
            f"{total_patches} patches -> {total_image_tokens} image tokens, pixel_dim={pixel_dim}"
        )

        def run_once():
            out = model(pixel_values, grid_thw)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        logger.info(f"Warmup ({args.warmup} iters, compiles programs)...")
        for i in range(args.warmup):
            t0 = time.time()
            run_once()
            logger.info(f"  warmup {i + 1}/{args.warmup}: {time.time() - t0:.3f}s")

        logger.info(f"Timing ({args.iters} iters)...")
        latencies = []
        for i in range(args.iters):
            t0 = time.time()
            out = model(pixel_values, grid_thw)
            ttnn.synchronize_device(mesh_device)
            dt = time.time() - t0
            ttnn.deallocate(out)
            latencies.append(dt)
            logger.info(f"  iter {i + 1}/{args.iters}: {dt * 1e3:.2f} ms")

        _report(latencies, total_patches, total_image_tokens, len(grids), depth, rows, cols, args.dtype)

    finally:
        ttnn.close_mesh_device(mesh_device)
        if multi:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _report(latencies, patches, image_tokens, num_images, depth, rows, cols, dtype):
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    std = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    best = min(latencies)
    worst = max(latencies)

    line = "=" * 64
    print(f"\n{line}")
    print("  DropInVisionTransformer benchmark")
    print(line)
    print(f"  mesh           : {rows}x{cols} ({'TP' if rows * cols > 1 else 'single device'})")
    print(f"  vision depth   : {depth} layers")
    print(f"  dtype          : {dtype}")
    print(f"  images / call  : {num_images}  ({patches} patches, {image_tokens} image tokens)")
    print(f"  iterations     : {len(latencies)}")
    print(line)
    print(f"  latency mean   : {mean * 1e3:8.2f} ms")
    print(f"  latency median : {median * 1e3:8.2f} ms")
    print(f"  latency stdev  : {std * 1e3:8.2f} ms")
    print(f"  latency best   : {best * 1e3:8.2f} ms")
    print(f"  latency worst  : {worst * 1e3:8.2f} ms")
    print(line)
    print(f"  throughput     : {patches / mean:10.1f} patches/s")
    print(f"                 : {image_tokens / mean:10.1f} image-tokens/s")
    print(f"                 : {num_images / mean:10.2f} images/s")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
