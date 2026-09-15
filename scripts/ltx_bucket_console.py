#!/usr/bin/env python3
"""Interactive LTX-2.3 bucket console for a 4x8 Blackhole Galaxy."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import torch
import ttnn
from loguru import logger

from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    LTX_BUCKET_LADDER,
    LTX_CANVASES,
    LTX_DURATION_VALUES,
    LTX_FAST_AUDIO_N_BUCKET,
    LTX_FAST_1080P_25FPS_6S_LADDER,
    LTX_FPS_VALUES,
    LTX_SERVED_CANVASES,
    default_ltx_checkpoint,
    default_ltx_gemma,
    ltx_aligned_num_frames,
    ltx_served_configs,
    route_ltx_config,
)
from models.tt_dit.utils.test import create_fabric_router_config
from models.tt_dit.utils.tracing import Tracer

ELEVEN_BUCKET_LADDER = (
    8704,
    12288,
    17408,
    24576,
    34560,
    48384,
    67840,
    94976,
    133120,
    186368,
    261120,
)

T = TypeVar("T")


class Fast1080pLTXPipeline(LTXDistilledPipeline):
    bucket_ladder = LTX_FAST_1080P_25FPS_6S_LADDER
    audio_n_bucket = LTX_FAST_AUDIO_N_BUCKET


class ElevenBucketLTXPipeline(LTXDistilledPipeline):
    bucket_ladder = ELEVEN_BUCKET_LADDER


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("fast-1080p6", "wide-6", "wide-11"),
        default="fast-1080p6",
        help="trace ladder profile (default: fast-1080p6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/ltx_bucket_console"),
        help="directory for generated MP4 files",
    )
    parser.add_argument(
        "--trace-region-size",
        type=int,
        default=int(os.environ.get("LTX_BUCKET_TRACE_REGION_SIZE", "1600000000")),
        help="per-device trace-region reservation in bytes",
    )
    parser.add_argument(
        "--component-traces",
        choices=("dit", "audio", "vae", "all"),
        default=os.environ.get("LTX_COMPONENT_TRACES", "dit"),
        help="resident trace experiment stage: dit; +audio; +video VAE; or +upsampler (default: dit)",
    )
    parser.add_argument(
        "--device-rope",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LTX_DEVICE_ROPE_MATERIALIZE", "0") in ("1", "true", "True"),
        help="expand compact video RoPE tables on device (default: off until parity validation)",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="exit after trace capture and residency reporting without entering the request loop",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run every served configuration non-interactively and save replay/parity metrics",
    )
    parser.add_argument(
        "--matrix-repeats",
        type=int,
        default=2,
        help="generation count per matrix configuration (default: 2)",
    )
    return parser.parse_args()


def _select(label: str, options: tuple[T, ...]) -> T | None:
    print(f"\n{label}:")
    for index, option in enumerate(options, 1):
        if isinstance(option, str) and option in LTX_CANVASES:
            height, width = LTX_CANVASES[option]
            display = f"{option} ({width}x{height})"
        else:
            display = str(option)
        print(f"  {index}. {display}")

    while True:
        answer = input(f"Select {label.lower()} [1-{len(options)}], or q to quit: ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        try:
            index = int(answer)
        except ValueError:
            index = -1
        if 1 <= index <= len(options):
            return options[index - 1]
        for option in options:
            if answer == str(option):
                return option
        print("Invalid selection.")


def _set_fabric() -> None:
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(8192),
    )


def _open_mesh(trace_region_size: int) -> tuple[ttnn.MeshDevice, ttnn.MeshDevice]:
    _set_fabric()
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(4, 8),
        trace_region_size=trace_region_size,
        l1_small_size=32768,
    )
    return parent, parent.create_submesh(ttnn.MeshShape(4, 8))


def _close_mesh(parent: ttnn.MeshDevice | None, mesh: ttnn.MeshDevice | None) -> None:
    if mesh is not None:
        ttnn.close_mesh_device(mesh)
    if parent is not None:
        ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _config_fits(ladder: tuple[int, ...], canvas: str, fps: int, duration: int, *, audio_n_bucket: int) -> bool:
    try:
        route_ltx_config(canvas, fps, duration, ladder=ladder, audio_n_bucket=audio_n_bucket)
    except ValueError:
        return False
    return True


def _interactive_loop(
    pipeline: LTXDistilledPipeline,
    output_dir: Path,
    ladder: tuple[int, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_number = 0

    print("\nLTX console ready. Traces remain resident until you quit.")
    print(f"Warm rungs: {sorted(pipeline._warm_rungs)}")
    print(f"Audio bucket: {pipeline.audio_n_bucket}")

    while True:
        canvas = _select("Resolution", LTX_SERVED_CANVASES)
        if canvas is None:
            return
        supported_fps = tuple(
            fps
            for fps in LTX_FPS_VALUES
            if any(
                _config_fits(
                    ladder,
                    canvas,
                    fps,
                    duration,
                    audio_n_bucket=pipeline.audio_n_bucket,
                )
                for duration in LTX_DURATION_VALUES
            )
        )
        fps = _select("FPS", supported_fps)
        if fps is None:
            return
        supported_durations = tuple(
            duration
            for duration in LTX_DURATION_VALUES
            if _config_fits(
                ladder,
                canvas,
                fps,
                duration,
                audio_n_bucket=pipeline.audio_n_bucket,
            )
        )
        if not supported_durations:
            print(f"No supported durations for {canvas} at {fps} FPS under this profile.")
            continue
        duration = _select("Duration (seconds)", supported_durations)
        if duration is None:
            return

        prompt = input("Prompt (blank uses the default): ").strip() or DEFAULT_LTX_PROMPT
        height, width = LTX_CANVASES[canvas]
        num_frames = ltx_aligned_num_frames(fps, duration)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"{timestamp}_{request_number:04d}_{canvas}_{fps}fps_{duration}s.mp4"

        print(f"\nGenerating {canvas}, {fps} FPS, {duration}s " f"({num_frames} aligned frames) -> {output_path}")
        started = time.time()
        try:
            pipeline.generate(
                prompt,
                output_path=str(output_path),
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                seed=10 + request_number,
            )
        except Exception:
            logger.exception("Generation failed; the console will remain open if the runtime is recoverable")
        else:
            print(f"Done in {time.time() - started:.1f}s: {output_path}")
        request_number += 1


def _sample_tensor(value: torch.Tensor, max_samples: int = 262_144) -> torch.Tensor:
    flat = torch.as_tensor(value).detach().float().reshape(-1)
    stride = max(1, (flat.numel() + max_samples - 1) // max_samples)
    return flat[::stride][:max_samples].clone()


def _pcc(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.shape != rhs.shape:
        return float("nan")
    lhs = lhs.double()
    rhs = rhs.double()
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    denominator = lhs.norm() * rhs.norm()
    if denominator == 0:
        return 1.0 if torch.equal(lhs, rhs) else 0.0
    return float(torch.dot(lhs, rhs) / denominator)


def _matrix_loop(
    pipeline: LTXDistilledPipeline,
    output_dir: Path,
    served_configs: tuple[tuple[str, int, int], ...],
    repeats: int,
    component_traces: str,
) -> None:
    if repeats < 2:
        raise ValueError("--matrix-repeats must be at least 2 to validate replay parity")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_residency = Tracer.residency(pipeline.mesh_device)
    baseline_addresses = Tracer.input_buffer_addresses(pipeline.mesh_device)
    results = []
    failures = []

    for config_index, (canvas, fps, duration) in enumerate(served_configs, 1):
        height, width = LTX_CANVASES[canvas]
        num_frames = ltx_aligned_num_frames(fps, duration)
        print(
            f"\nMatrix {config_index}/{len(served_configs)}: " f"{canvas}, {fps} FPS, {duration}s ({num_frames} frames)"
        )
        reference_video = None
        reference_audio = None
        runs = []
        try:
            for repeat in range(repeats):
                started = time.perf_counter()
                video, audio = pipeline.generate(
                    DEFAULT_LTX_PROMPT,
                    output_path=None,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    seed=10 + config_index,
                    output_type="float",
                )
                elapsed = time.perf_counter() - started
                video_sample = _sample_tensor(video)
                audio_sample = _sample_tensor(audio.waveform)
                run = {
                    "repeat": repeat + 1,
                    "seconds": elapsed,
                    "video_std": float(torch.as_tensor(video).float().std()),
                    "audio_std": float(torch.as_tensor(audio.waveform).float().std()),
                    "stage_seconds": dict(pipeline.last_timings),
                }
                if reference_video is None:
                    reference_video, reference_audio = video_sample, audio_sample
                else:
                    run["video_pcc_vs_first"] = _pcc(reference_video, video_sample)
                    run["audio_pcc_vs_first"] = _pcc(reference_audio, audio_sample)
                    if run["video_pcc_vs_first"] < 0.999 or run["audio_pcc_vs_first"] < 0.999:
                        raise RuntimeError(
                            f"replay parity failed: video PCC={run['video_pcc_vs_first']:.6f}, "
                            f"audio PCC={run['audio_pcc_vs_first']:.6f}"
                        )
                runs.append(run)
                del video, audio, video_sample, audio_sample
                gc.collect()

                if Tracer.residency(pipeline.mesh_device) != baseline_residency:
                    raise RuntimeError("resident trace count/bytes changed while serving")
                if Tracer.input_buffer_addresses(pipeline.mesh_device) != baseline_addresses:
                    raise RuntimeError("captured input buffer address changed while serving")

            results.append(
                {
                    "canvas": canvas,
                    "fps": fps,
                    "duration": duration,
                    "num_frames": num_frames,
                    "runs": runs,
                }
            )
        except Exception as exc:
            logger.exception(f"matrix configuration failed: {canvas} {fps}fps {duration}s")
            failures.append(
                {
                    "canvas": canvas,
                    "fps": fps,
                    "duration": duration,
                    "error": repr(exc),
                }
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"{timestamp}_trace_matrix_{component_traces}.json"
    report_path.write_text(
        json.dumps(
            {
                "component_traces": component_traces,
                "trace_count": baseline_residency[0],
                "trace_bytes": baseline_residency[1],
                "config_count": len(served_configs),
                "results": results,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Matrix report: {report_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(served_configs)} matrix configurations failed")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    trace_audio = args.component_traces in ("audio", "vae", "all")
    trace_video_vae = args.component_traces in ("vae", "all")
    trace_upsampler = args.component_traces == "all"
    os.environ["LTX_VOC_TRACE"] = "1" if trace_audio else "0"
    os.environ["LTX_BWE_TRACE"] = "1" if trace_audio else "0"
    os.environ["LTX_MEL_TRACE"] = "1" if trace_audio else "0"
    os.environ["LTX_VIDEO_VAE_TRACE"] = "1" if trace_video_vae else "0"
    os.environ["LTX_UPSAMPLER_TRACE"] = "1" if trace_upsampler else "0"
    os.environ["LTX_DEVICE_ROPE_MATERIALIZE"] = "1" if args.device_rope else "0"
    # Legacy combined gate remains off: explicit component gates above avoid coupling video and mel VAE.
    os.environ["LTX_VAE_TRACE"] = "0"
    os.environ.setdefault("LTX_VAE_TEMPORAL_CHUNK_LATENTS", "0")

    profiles = {
        "fast-1080p6": (Fast1080pLTXPipeline, LTX_FAST_1080P_25FPS_6S_LADDER),
        "wide-6": (LTXDistilledPipeline, LTX_BUCKET_LADDER),
        "wide-11": (ElevenBucketLTXPipeline, ELEVEN_BUCKET_LADDER),
    }
    pipeline_class, ladder = profiles[args.profile]
    audio_n_bucket = pipeline_class.audio_n_bucket
    served_configs = tuple(
        config for config in ltx_served_configs() if _config_fits(ladder, *config, audio_n_bucket=audio_n_bucket)
    )
    postprocess_shapes = {
        (ltx_aligned_num_frames(fps, duration), *LTX_CANVASES[canvas]) for canvas, fps, duration in served_configs
    }
    os.environ["LTX_SERVED_CONFIGS"] = ",".join(
        f"{canvas}:{fps}:{duration}" for canvas, fps, duration in served_configs
    )
    parent = None
    mesh = None
    pipeline = None
    try:
        parent, mesh = _open_mesh(args.trace_region_size)
        print(f"Warming {len(ladder)} resident DiT traces: {ladder}")
        print(f"Audio bucket: {audio_n_bucket}")
        print(f"Component trace stage: {args.component_traces}")
        print(f"Profile {args.profile} serves {len(served_configs)} resolution/FPS/duration combinations.")
        print(f"Precompiling {len(postprocess_shapes)} exact upsampler/VAE shapes before trace capture.")
        print("The interactive loop starts only after the complete serving envelope is warm.")
        pipeline = pipeline_class.create_pipeline(
            mesh_device=mesh,
            checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
            gemma_path=default_ltx_gemma(),
            sp_axis=1,
            tp_axis=0,
            num_links=2,
            dynamic_load=False,
            topology=ttnn.Topology.Ring,
            is_fsdp=False,
            run_warmup=True,
            traced=True,
            num_frames=ltx_aligned_num_frames(25, 6),
            height=LTX_CANVASES["1080p-landscape"][0],
            width=LTX_CANVASES["1080p-landscape"][1],
            fps=25,
        )
        if pipeline._warm_rungs != frozenset(ladder):
            raise RuntimeError(f"expected warm rungs {ladder}, got {sorted(pipeline._warm_rungs)}")
        if args.warmup_only:
            print("Warmup-only trace fit completed.")
            return 0
        if args.matrix:
            _matrix_loop(
                pipeline,
                output_dir,
                served_configs,
                args.matrix_repeats,
                args.component_traces,
            )
            return 0
        _interactive_loop(pipeline, output_dir, ladder)
        return 0
    except KeyboardInterrupt:
        print("\nStopping LTX console.")
        return 130
    finally:
        if pipeline is not None:
            pipeline.release_traces()
        pipeline = None
        gc.collect()
        _close_mesh(parent, mesh)


if __name__ == "__main__":
    sys.exit(main())
