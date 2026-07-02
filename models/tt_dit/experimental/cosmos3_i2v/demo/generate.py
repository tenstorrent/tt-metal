# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone CLI to generate a Cosmos3-Super-Image2Video clip and save MP4.

Builds the native (or native-cfg) pipeline and opens the mesh device itself so
you don't need pytest in the loop. All knobs are command-line flags.

Examples (run from tt-metal root, with the venv activated):

    # LoudBox / T3K (8 chips, BFP8 auto-selected):
    python -m models.tt_dit.experimental.cosmos3_i2v.demo.generate \\
        --image ~/cat.jpg \\
        --prompt "a cat walks across a sunlit kitchen floor" \\
        --steps 50 \\
        --out /tmp/cat_walk.mp4

    # BH Galaxy (32 chips, bfloat16):
    python -m models.tt_dit.experimental.cosmos3_i2v.demo.generate \\
        --image ~/landscape.jpg \\
        --prompt "drone footage of a misty mountain valley at dawn" \\
        --steps 30 \\
        --frames 33 --height 512 --width 512 \\
        --out /tmp/valley.mp4

`--mesh-shape auto` (default) maps available device counts to known mesh
layouts. `--weight-dtype auto` (default) picks BFP8 on a (1, 8) WH mesh
and bfloat16 elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Silence the HF tokenizers fork warning. transformers' fast tokenizer spins up a
# thread pool on first use; later in the pipeline the MP4 writer forks (ffmpeg
# subprocess), which prints "process just got forked, after parallelism has
# already been used. Disabling parallelism to avoid deadlocks". Cosmetic — set
# before `transformers` is imported anywhere downstream.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a Cosmos3-Super-Image2Video clip on Tenstorrent hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to the reference image (JPEG/PNG). Omit for text-to-video (no conditioning frame).",
    )
    p.add_argument("--prompt", required=True, help="Text prompt for the diffusion model.")
    p.add_argument(
        "--negative-prompt",
        default=None,
        help=(
            "Classifier-free-guidance negative prompt. Omit to use the mode's recommended "
            "default (the NVIDIA quality-control string for video modes; empty for text2image)."
        ),
    )
    p.add_argument("--steps", type=int, default=35, help="UniPC denoise steps. NVIDIA I2V framework default.")
    p.add_argument(
        "--frames", type=int, default=17, help="Number of output frames. Must be 4k+1 (TI2V VAE constraint)."
    )
    p.add_argument("--height", type=int, default=256, help="Output frame height in pixels.")
    p.add_argument("--width", type=int, default=256, help="Output frame width in pixels.")
    p.add_argument("--fps", type=int, default=16, help="MP4 framerate (matches the Cosmos3 base_fps default).")
    p.add_argument("--out", type=Path, default=Path("cosmos3_i2v.mp4"), help="Output MP4 path.")
    p.add_argument(
        "--mesh-shape",
        default="auto",
        help="'auto' (pick from device count), or 'RxC' (e.g. '1x8', '4x8').",
    )
    p.add_argument(
        "--weight-dtype",
        default="auto",
        choices=["auto", "bfloat16", "bfloat8_b"],
        help="On-device weight precision. 'auto' picks BFP8 for (1, 8) meshes, bfloat16 otherwise.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generator seed. Default: nondeterministic.",
    )
    p.add_argument(
        "--output-type",
        default="video",
        choices=["video", "latent"],
        help=(
            "Diagnostic. 'video' (default) runs the full pipeline including VAE decode and "
            "writes an MP4. 'latent' returns the post-denoise latent tensor without VAE decode; "
            "stats (min/max/mean/std/nan-count) are printed and the tensor is dumped to "
            "`<out>.latent.pt` so you can inspect what the trunk produced when the video came out black."
        ),
    )
    p.add_argument(
        "--pipeline",
        default="native-cfg",
        choices=["native-cfg"],
        help=(
            "Pipeline factory. 'native-cfg' runs the 64-layer trunk + final RMSNorms on "
            "device, splits the mesh's smaller axis into two submeshes, and runs the "
            "cond/uncond passes concurrently. On a 1-axis mesh it transparently falls back "
            "to a single submesh. This is the only supported path — the single-mesh 'native' "
            "trunk produces noise at sp_factor=4 on a full 4x8."
        ),
    )
    p.add_argument(
        "--no-tt-vae",
        action="store_true",
        help=(
            "Disable the TT-NN VAE adapters and fall back to host PyTorch VAE. Useful for "
            "bisecting whether a regression lives in the TT VAE vs the trunk or scheduler."
        ),
    )
    p.add_argument(
        "--flow-shift",
        type=float,
        default=6.0,
        help=(
            "Flow-matching sigma shift applied to the UniPC scheduler. Default 6.0 is the unified "
            "Cosmos3 preset (all modes). Cosmos3 paper Table 21 lists shift=5 for "
            "Cosmos3-Super-Image2Video and shift=10 for the Audio-Visual omni preset. "
            "σ' = s·σ / (1 + (s-1)·σ)."
        ),
    )
    p.add_argument(
        "--num-links",
        type=int,
        default=None,
        help=(
            "Override CCL num_links. Default = auto (2 on BH, 4 on WH-4x8, 1 elsewhere). "
            "Bump if you see throughput-limited links; reduce if you hit fabric errors."
        ),
    )
    p.add_argument(
        "--cfg-serial-dispatch",
        action="store_true",
        help=(
            "Diagnostic for --pipeline native-cfg: run the two trunks sequentially on the two submeshes "
            "instead of in parallel threads. Isolates correctness bugs in the threading layer vs the "
            "dual-submesh build."
        ),
    )
    p.add_argument(
        "--vae-decoder-t-chunk-size",
        type=int,
        default=4,
        help=(
            "Temporal chunk size for the TT VAE decoder. Default 4 fits the cfg-parallel submesh "
            "(half the SP factor); full-T (0/None) OOMs there at 720p/189f. Chunk caps harmlessly "
            "below T, so 4 is safe at any frame count."
        ),
    )
    p.add_argument(
        "--vae-encoder-t-chunk-size",
        type=int,
        default=4,
        help="Temporal chunk size for the TT VAE encoder. Default 4; same memory/perf trade as the decoder.",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help=(
            "Classifier-free guidance scale. Reference Cosmos3OmniPipeline default is 6.0. "
            "Lower (3-5) reduces temporal flicker / wall-color drift in I2V output at the cost "
            "of weaker text adherence. Higher (7-9) amplifies prompt match but adds wobble."
        ),
    )
    return p.parse_args(argv)


def resolve_mesh_shape(spec: str, available: int) -> tuple[int, int]:
    """Map --mesh-shape ('auto' or 'RxC') to a (rows, cols) tuple."""
    if spec == "auto":
        if available >= 32:
            return (4, 8)
        if available >= 8:
            return (1, 8)
        if available >= 4:
            return (1, 4)
        if available >= 2:
            return (1, 2)
        return (1, 1)
    try:
        rows, cols = (int(x) for x in spec.lower().split("x"))
    except ValueError as e:
        raise SystemExit(f"--mesh-shape must be 'auto' or 'RxC' (got {spec!r})") from e
    if rows * cols > available:
        raise SystemExit(f"mesh {rows}x{cols} ({rows * cols} chips) > available ({available})")
    return (rows, cols)


def resolve_weight_dtype(spec: str, mesh_shape: tuple[int, int]):
    """Map --weight-dtype to a ttnn dtype handle."""
    import ttnn

    if spec == "auto":
        return ttnn.bfloat8_b if mesh_shape == (1, 8) else ttnn.bfloat16
    if spec == "bfloat16":
        return ttnn.bfloat16
    return ttnn.bfloat8_b


def open_mesh(mesh_shape: tuple[int, int]):
    """Open a mesh device with FABRIC_1D config; caller closes via close_mesh()."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    trace_region_size = int(os.environ.get("TT_COSMOS3_TRACE_REGION_SIZE", str(200 * 1024 * 1024)))
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        trace_region_size=trace_region_size,
    )


def close_mesh(mesh) -> None:
    """Close submeshes, the parent mesh, then disable the fabric."""
    import ttnn

    for sub in mesh.get_submeshes():
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.image is not None and not args.image.exists():
        raise SystemExit(f"Reference image not found: {args.image}")
    if (args.frames - 1) % 4 != 0:
        raise SystemExit(
            f"--frames must be 4k+1 for the TI2V VAE (got {args.frames}). "
            f"Try {((args.frames - 1) // 4) * 4 + 1} or {((args.frames - 1) // 4 + 1) * 4 + 1}."
        )

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    import ttnn

    available = ttnn.get_num_devices()
    mesh_shape = resolve_mesh_shape(args.mesh_shape, available)
    weight_dtype = resolve_weight_dtype(args.weight_dtype, mesh_shape)

    print(
        f"[generate] image={args.image} prompt={args.prompt!r}\n"
        f"[generate] mesh={mesh_shape} (of {available} available), weight_dtype={weight_dtype}\n"
        f"[generate] steps={args.steps} frames={args.frames} size={args.width}x{args.height} fps={args.fps}\n"
        f"[generate] pipeline={args.pipeline} flow_shift={args.flow_shift}\n"
        f"[generate] out={args.out}",
        flush=True,
    )

    ref_image = None
    if args.image is not None:
        ref_image = Image.open(args.image).convert("RGB").resize((args.width, args.height))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    mesh = open_mesh(mesh_shape)
    try:
        t0 = time.time()
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native_cfg import (
            build_cosmos3_i2v_native_cfg_pipeline,
        )

        pipe = build_cosmos3_i2v_native_cfg_pipeline(
            mesh,
            dtype=torch.bfloat16,
            use_tt_vae=not args.no_tt_vae,
            num_links=args.num_links,
            flow_shift=args.flow_shift,
            trunk_weight_dtype=weight_dtype,
            vae_decoder_t_chunk_size=args.vae_decoder_t_chunk_size,
            vae_encoder_t_chunk_size=args.vae_encoder_t_chunk_size,
            serial_dispatch=args.cfg_serial_dispatch,
        )
        print(f"[generate] pipeline built and weights placed in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        print(
            f"[generate] sampler: steps={args.steps} flow_shift={args.flow_shift} "
            f"guidance_scale={args.guidance_scale}",
            flush=True,
        )
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.cosmos3_mode import run_cosmos3
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native import _make_release_callback

        call_kwargs = dict(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            output_type="latent" if args.output_type == "latent" else "pil",
            callback_on_step_end=_make_release_callback(args.steps),
            callback_on_step_end_tensor_inputs=[],
        )
        if generator is not None:
            call_kwargs["generator"] = generator
        result = run_cosmos3(
            pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=ref_image,
            num_frames=args.frames,
            **call_kwargs,
        )
        print(f"[generate] denoise + decode in {time.time() - t1:.1f}s", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.output_type == "latent":
            latent = result.video  # When output_type='latent', `.video` carries the raw latent.
            tensor = latent if hasattr(latent, "float") else torch.as_tensor(latent)
            float_view = tensor.detach().to(torch.float32).cpu()
            shape = tuple(float_view.shape)
            nan_count = int(torch.isnan(float_view).sum().item())
            inf_count = int(torch.isinf(float_view).sum().item())
            print(
                f"[generate] latent stats: shape={shape}, "
                f"min={float_view.min().item():.4f} max={float_view.max().item():.4f} "
                f"mean={float_view.mean().item():.4f} std={float_view.std().item():.4f} "
                f"nan={nan_count} inf={inf_count}",
                flush=True,
            )
            latent_path = args.out.with_suffix(args.out.suffix + ".latent.pt")
            torch.save(tensor.detach().cpu(), latent_path)
            print(f"[generate] wrote {latent_path} ({latent_path.stat().st_size / 1024:.1f} KB)")
        elif args.frames == 1:
            # text2image: the pipeline returns a one-frame clip; write it as a still
            # rather than muxing a degenerate single-frame MP4.
            image_path = args.out.with_suffix(".png") if args.out.suffix.lower() == ".mp4" else args.out
            result.video[0].save(image_path)
            size_kb = image_path.stat().st_size / 1024
            print(f"[generate] wrote {image_path} ({size_kb:.1f} KB)")
        else:
            export_to_video(result.video, str(args.out), fps=args.fps)
            size_kb = args.out.stat().st_size / 1024
            print(f"[generate] wrote {args.out} ({size_kb:.1f} KB)")
    finally:
        close_mesh(mesh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
