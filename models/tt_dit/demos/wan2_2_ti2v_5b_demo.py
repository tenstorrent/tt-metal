# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B demo: real prompt (and optionally a real image) -> TT execution -> mp4.

Text-to-video:

    python -m models.tt_dit.demos.wan2_2_ti2v_5b_demo \\
        --prompt "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage." \\
        --out /home/ttuser/wan5b_demo_t2v.mp4

Image-to-video (the image is resized to the target resolution and conditions frame 0):

    python -m models.tt_dit.demos.wan2_2_ti2v_5b_demo \\
        --image /path/to/frame.png --prompt "The camera slowly pushes in as the cat turns." \\
        --out /home/ttuser/wan5b_demo_i2v.mp4

Defaults match the perf-gated production configuration: single Blackhole Galaxy (4x8, ring),
1280x704, 81 frames, 40 steps, CFG 4.0, warm-traced. `--steps 2 --frames 21 --untraced` is a
quick smoke. Environment: run from the tt-metal root with `TT_METAL_HOME`, `PYTHONPATH` and
`HF_HOME` set the way the 5B tests are run; `WAN5B_QUANT_CONFIG` is honoured (opt-in presets).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.events import log_event_section
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b_i2v import WanTI2V5BI2VPipeline
from models.tt_dit.pipelines.wan.quant_config import (
    configure_trace_mode_from_env,
    device_params_for_trace_mode,
    set_quant_config_from_env,
)

MESH_SHAPE = (4, 8)
TRACE_REGION_SIZE = 150_000_000  # same as the 5B perf tests (DEVICE_PARAMS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, help="text prompt")
    p.add_argument("--negative-prompt", default=None, help="negative prompt (default: the pipeline's)")
    p.add_argument("--image", default=None, help="conditioning image; switches the demo to image-to-video")
    p.add_argument("--out", default="/home/ttuser/wan5b_demo.mp4", help="output mp4 path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704, help="must be a multiple of 32 (704, not 720)")
    p.add_argument("--frames", type=int, default=81, help="4k+1 frames (81, 121, 21 ...)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--untraced", action="store_true", help="eager execution (slower; skips trace capture)")
    p.add_argument("--repeat", type=int, default=1, help="extra timed generations after the first (traced only)")
    return p.parse_args(argv)


def open_mesh() -> ttnn.MeshDevice:
    """Open the 4x8 Blackhole Galaxy with the 1D ring fabric the 5B pipeline is tuned for."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    params = device_params_for_trace_mode({"trace_region_size": TRACE_REGION_SIZE})
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), **params)


def close_mesh(mesh: ttnn.MeshDevice) -> None:
    for sub in mesh.get_submeshes():
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def save_outputs(frames_u8: np.ndarray, out_path: str, fps: int) -> None:
    """PNG previews always; mp4 when an ffmpeg is reachable (imageio_ffmpeg on PYTHONPATH)."""
    from PIL import Image

    base = os.path.splitext(out_path)[0]
    t = frames_u8.shape[0]
    for tag, idx in (("first", 0), ("mid", t // 2), ("last", t - 1)):
        Image.fromarray(frames_u8[idx]).save(f"{base}_{tag}.png")
    logger.info(f"Saved preview PNGs: {base}_{{first,mid,last}}.png")
    try:
        from models.tt_dit.utils.video import export_to_video

        export_to_video(frames_u8, out_path, fps=fps)
        logger.info(f"Saved video: {out_path} ({t} frames @ {fps} fps)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"mp4 export failed ({e!r}); the PNG previews are the output")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.height % 32 or args.width % 32:
        logger.error("height and width must be multiples of 32 (the 16x VAE and patch_size=2)")
        return 2
    if (args.frames - 1) % 4:
        logger.error("frames must be 4k+1 (21, 81, 121, ...)")
        return 2
    if not ttnn.device.is_blackhole():
        logger.error("the TI2V-5B pipeline targets a Blackhole Galaxy")
        return 2

    image = None
    if args.image is not None:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        logger.info(f"conditioning image: {args.image} {image.size} -> resized to {args.width}x{args.height}")

    traced = not args.untraced
    pipeline_cls = WanTI2V5BI2VPipeline if image is not None else WanTI2V5BPipeline
    mode = "I2V" if image is not None else "T2V"
    logger.info(
        f"Wan2.2 TI2V-5B {mode}: {args.width}x{args.height}, {args.frames} frames, {args.steps} steps, "
        f"seed {args.seed}, {'traced' if traced else 'eager'}"
    )

    mesh = open_mesh()
    try:
        t0 = time.perf_counter()
        pipeline = pipeline_cls.create_pipeline(
            mesh_device=mesh, height=args.height, width=args.width, num_frames=args.frames, run_warmup=True
        )
        quant = set_quant_config_from_env(pipeline, rewarm=traced)
        trace_mode = configure_trace_mode_from_env(pipeline)
        logger.info(
            f"pipeline ready in {time.perf_counter() - t0:.1f}s (quant preset: {quant or 'none'}, trace mode: {trace_mode})"
        )

        call_kwargs = dict(
            prompts=[args.prompt],
            negative_prompts=[args.negative_prompt] if args.negative_prompt else None,
            image_prompt=image,
            num_inference_steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=None,
            output_type="uint8",
            traced=traced,
            on_event=log_event_section,
        )

        frames = None
        n_runs = 1 + (args.repeat if traced else 0)
        for i in range(n_runs):
            t1 = time.perf_counter()
            with torch.no_grad():
                frames = pipeline(**call_kwargs)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t1
            tag = "cold (trace capture)" if traced and i == 0 else ("warm traced" if traced else "eager")
            logger.info(
                f"generation {i + 1}/{n_runs} [{tag}]: {dt:.2f}s, {dt / args.steps * 1000:.0f} ms/step incl. encode+decode"
            )

        if hasattr(pipeline, "release_traces"):
            pipeline.release_traces()

        frames_u8 = np.asarray(frames[0]).astype(np.uint8)  # (T, H, W, 3)
        assert frames_u8.shape == (args.frames, args.height, args.width, 3), frames_u8.shape
        if int(ttnn.distributed_context_get_rank()) == 0:
            save_outputs(frames_u8, args.out, args.fps)
    finally:
        close_mesh(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
