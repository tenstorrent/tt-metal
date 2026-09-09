# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone latency/throughput benchmark for the Gemma-4 TT vision encoder.

Drives ``VisionTransformer`` the same way as
``models/demos/gemma4/tests/unit/test_vision_encoder.py``: image-processor
pixels on device, patch embed + 2D RoPE + transformer blocks. No text model,
no PCC in the timed path.

Examples
--------
BH Galaxy (8, 4), dummy weights, full depth (same fabric as the encoder unit test)::

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=BHGLX \\
        python models/demos/gemma4/demo/benchmark_vision.py

Quick iteration (2 layers, fewer iters)::

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=BHGLX \\
        python models/demos/gemma4/demo/benchmark_vision.py --num-layers 2 --iters 5

Explicit mesh / 300-DPI scanned-letter image (the unit-test default)::

    python models/demos/gemma4/demo/benchmark_vision.py \\
        --mesh-shape 8 4 --preset 300dpi --iters 10

Tracy device-op capture of the timed region::

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -- \\
        models/demos/gemma4/demo/benchmark_vision.py --num-layers 2 --tracy
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torchvision.transforms as T
from loguru import logger
from transformers import Gemma4ImageProcessor

import ttnn
from models.demos.gemma4.tests.unit.test_vision_attention import convert_vision_block_hf_to_meta
from models.demos.gemma4.tt.vision.vision_encoder import VisionTransformer
from models.demos.gemma4.tt.vision.vision_model_config import (
    VISION_MESH_DEVICE_MAP,
    VisionModelArgs,
    vision_mesh_shape_from_env,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal

# Same image grids as test_vision_encoder.py (C, H_patches, W_patches). Pixel
# spatial size is patch_size (16) times the patch grid.
_PRESETS = {
    "300dpi": {
        "token_budget": 280 * 9,
        "image_grid_chw": [3, 110, 85],
    },
    "240dpi": {
        "token_budget": 560 * 9,
        "image_grid_chw": [3, 66, 54],
    },
}

_L1_SMALL_SIZE = int(os.environ.get("GEMMA4_L1_SMALL_SIZE", "24576"))


def _parse_grid(s: str):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--grid expects 'c,h,w' patch grid, got '{s}'")
    return [int(p) for p in parts]


def _resolve_mesh_shape(cli_shape):
    if cli_shape is not None:
        return tuple(cli_shape)
    shape = vision_mesh_shape_from_env()
    if isinstance(shape, int):
        return (1, max(shape, 1))
    return tuple(shape)


def _fabric_config(rows: int, cols: int):
    """Match the encoder unit test: 2D torus on a 2D mesh, 1D fabric on a line."""
    if rows > 1 and cols > 1:
        return ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    if rows * cols > 1:
        return ttnn.FabricConfig.FABRIC_1D
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mesh-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Mesh (DP, TP). Default: MESH_DEVICE env via VISION_MESH_DEVICE_MAP "
        f"(known: {', '.join(sorted(VISION_MESH_DEVICE_MAP))}).",
    )
    p.add_argument(
        "--preset",
        choices=sorted(_PRESETS),
        default="300dpi",
        help="Image size from the encoder unit test. Default: 300dpi.",
    )
    p.add_argument(
        "--grid",
        type=_parse_grid,
        default=None,
        help="Override preset patch grid as 'c,h,w' (e.g. 3,110,85).",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="max_soft_tokens * 9 passed to the image processor. Default: preset.",
    )
    p.add_argument("--num-layers", type=int, default=None, help="Override vision depth. Default: full HF depth.")
    p.add_argument("--iters", type=int, default=10, help="Timed iterations. Default: 10.")
    p.add_argument("--warmup", type=int, default=2, help="Untimed compile/cache iterations. Default: 2.")
    p.add_argument(
        "--dtype",
        choices=["bfp8", "bf16"],
        default="bfp8",
        help="Vision weight/compute dtype. Default: bfp8 (matches the unit test).",
    )
    p.add_argument(
        "--real-weights",
        action="store_true",
        help="Load the HF checkpoint instead of dummy weights (compute is unchanged).",
    )
    p.add_argument(
        "--tracy",
        action="store_true",
        help="Wrap timed iterations in tracy signposts start/stop.",
    )
    return p.parse_args()


def _build_host_inputs(model_args, image_grid_chw, token_budget):
    random_img = torch.rand(image_grid_chw[0], image_grid_chw[1] * 16, image_grid_chw[2] * 16)
    img = T.ToPILImage()(random_img)
    image_processor = Gemma4ImageProcessor.from_pretrained(f"google/{model_args.model_name}")
    processed = image_processor(images=[img], max_soft_tokens=token_budget // 9, return_tensors="pt")
    pixel_position_ids = processed["image_position_ids"]
    pt_pixel_values = processed["pixel_values"]
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    return pt_pixel_values, pixel_position_ids, padding_positions


def _to_device(mesh_device, pt_pixel_values, pixel_position_ids, padding_positions):
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_pixel_values = ttnn.from_torch(
        pt_pixel_values.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_position_ids = ttnn.from_torch(
        pixel_position_ids.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    tt_padding_positions = ttnn.from_torch(
        padding_positions.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    return tt_pixel_values, tt_position_ids, tt_padding_positions


def _maybe_signpost():
    try:
        from tracy import signpost
    except ModuleNotFoundError:

        def signpost(*_a, **_k):
            pass

    return signpost


def main():
    args = parse_args()
    rows, cols = _resolve_mesh_shape(args.mesh_shape)
    dtype = ttnn.bfloat8_b if args.dtype == "bfp8" else ttnn.bfloat16
    preset = _PRESETS[args.preset]
    image_grid_chw = args.grid if args.grid is not None else preset["image_grid_chw"]
    token_budget = args.token_budget if args.token_budget is not None else preset["token_budget"]
    seq_len = ((token_budget // 2048) + 1) * 2048
    fabric = _fabric_config(rows, cols)

    logger.info(f"Opening mesh device {rows}x{cols} (DP={rows}, TP={cols}), fabric={fabric}")
    if fabric is not None:
        ttnn.set_fabric_config(fabric)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        l1_small_size=_L1_SMALL_SIZE,
    )
    signpost = _maybe_signpost() if args.tracy else None

    try:
        try:
            mesh_device.enable_program_cache()
        except AttributeError:
            pass

        model_args = VisionModelArgs(
            mesh_device,
            dummy_weights=not args.real_weights,
            max_batch_size=1,
            max_seq_len=seq_len,
        )
        logger.info(f"ccl_topology={model_args.ccl_topology()}")

        if args.num_layers:
            model_args.hf_config.vision_config.num_hidden_layers = args.num_layers
            from transformers import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        num_layers = model_args.hf_config.vision_config.num_hidden_layers

        logger.info(f"Loading HF vision tower (depth={num_layers}, dummy={not args.real_weights})...")
        t0 = time.perf_counter()
        reference_model = model_args.reference_vision_model(depth=num_layers)
        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_vision_block_hf_to_meta(
            state_dict, model_args.n_heads, model_args.n_kv_heads, model_args.head_dim
        )
        prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{prefix}.{k}": v for k, v in state_dict.items()}
        logger.info(f"Reference weights ready in {time.perf_counter() - t0:.1f}s")

        logger.info("Building TT VisionTransformer...")
        t0 = time.perf_counter()
        tt_model = VisionTransformer(
            args=model_args,
            tt_ccl=TT_CCL(mesh_device),
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )
        logger.info(f"TT encoder built in {time.perf_counter() - t0:.1f}s")

        pt_pixel_values, pixel_position_ids, padding_positions = _build_host_inputs(
            model_args, image_grid_chw, token_budget
        )
        num_patches = pixel_position_ids.shape[1]
        valid_patches = int((~padding_positions).sum().item())
        tt_pixel_values, tt_position_ids, tt_padding_positions = _to_device(
            mesh_device, pt_pixel_values, pixel_position_ids, padding_positions
        )
        logger.info(
            f"Inputs: grid={image_grid_chw}, token_budget={token_budget}, "
            f"num_patches={num_patches} (valid={valid_patches}), padded seq_len={seq_len}"
        )

        def run_once():
            out = tt_model(
                tt_pixel_values,
                tt_position_ids,
                tt_padding_positions,
                unpadded_seq_len=num_patches,
                seq_len=seq_len,
            )
            ttnn.synchronize_device(mesh_device)
            return out

        logger.info(f"Warmup ({args.warmup} iters, compiles programs)...")
        for i in range(args.warmup):
            t0 = time.perf_counter()
            out = run_once()
            ttnn.deallocate(out)
            logger.info(f"  warmup {i + 1}/{args.warmup}: {time.perf_counter() - t0:.3f}s")

        if signpost is not None:
            signpost("start")

        logger.info(f"Timing ({args.iters} iters)...")
        latencies = []
        for i in range(args.iters):
            t0 = time.perf_counter()
            out = run_once()
            dt = time.perf_counter() - t0
            ttnn.deallocate(out)
            latencies.append(dt)
            logger.info(f"  iter {i + 1}/{args.iters}: {dt * 1e3:.2f} ms")

        if signpost is not None:
            signpost("stop")

        _report(
            latencies,
            rows=rows,
            cols=cols,
            num_layers=num_layers,
            dtype=args.dtype,
            num_patches=num_patches,
            valid_patches=valid_patches,
            seq_len=seq_len,
            preset=args.preset,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _report(latencies, *, rows, cols, num_layers, dtype, num_patches, valid_patches, seq_len, preset):
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    std = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    best = min(latencies)
    worst = max(latencies)
    n_images = 1

    line = "=" * 64
    print(f"\n{line}")
    print("  Gemma-4 VisionTransformer benchmark")
    print(line)
    print(f"  mesh           : {rows}x{cols}  (DP={rows}, TP={cols})")
    print(f"  vision depth   : {num_layers} layers")
    print(f"  dtype          : {dtype}")
    print(f"  preset         : {preset}")
    print(f"  patches        : {num_patches}  (valid={valid_patches}, padded seq={seq_len})")
    print(f"  iterations     : {len(latencies)}")
    print(line)
    print(f"  latency mean   : {mean * 1e3:8.2f} ms")
    print(f"  latency median : {median * 1e3:8.2f} ms")
    print(f"  latency stdev  : {std * 1e3:8.2f} ms")
    print(f"  latency best   : {best * 1e3:8.2f} ms")
    print(f"  latency worst  : {worst * 1e3:8.2f} ms")
    print(line)
    print(f"  throughput     : {num_patches / mean:10.1f} patches/s")
    print(f"                 : {valid_patches / mean:10.1f} valid-patches/s")
    print(f"                 : {n_images / mean:10.2f} images/s")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
