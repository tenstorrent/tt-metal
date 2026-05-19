# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio

from models.experimental.audiox.demo.demo import _HF_CONFIG, run_demo


def _infer_conditioning_mode(
    prompt: str,
    *,
    video_path: Path | None,
    image_path: Path | None,
    audio_path: Path | None,
) -> str:
    has_text = bool(prompt)
    has_visual = video_path is not None or image_path is not None
    has_audio = audio_path is not None
    if has_visual and has_text:
        return "visual+text-to-audio"
    if has_visual:
        return "video-to-audio" if video_path is not None else "image-to-audio"
    if has_audio and has_text:
        return "audio+text-to-audio"
    if has_audio:
        return "audio-conditioned"
    return "text-to-audio"


def _summarize_audio_file(path: Path) -> dict:
    info = torchaudio.info(str(path))
    duration_seconds = 0.0 if info.sample_rate == 0 else info.num_frames / info.sample_rate
    return {
        "path": str(path),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "num_channels": info.num_channels,
        "duration_seconds": duration_seconds,
        "valid_16khz": info.sample_rate == _HF_CONFIG["output_sample_rate"],
    }


def _compare_audio_files(reference_path: Path, candidate_path: Path) -> dict:
    ref_audio, ref_sample_rate = torchaudio.load(str(reference_path))
    cand_audio, cand_sample_rate = torchaudio.load(str(candidate_path))
    same_sample_rate = ref_sample_rate == cand_sample_rate
    same_shape = tuple(ref_audio.shape) == tuple(cand_audio.shape)
    if not same_sample_rate or not same_shape:
        return {
            "same_sample_rate": same_sample_rate,
            "same_shape": same_shape,
            "mae": None,
            "max_abs": None,
        }

    delta = (ref_audio - cand_audio).abs()
    return {
        "same_sample_rate": True,
        "same_shape": True,
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
    }


def _build_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "cpu_reference.wav",
        output_dir / "tt_output.wav",
        output_dir / "validation_report.json",
    )


def _run_cpu_reference(args: argparse.Namespace, cpu_output: Path) -> dict:
    started_at = time.perf_counter()
    output_path = run_demo(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        output=cpu_output,
        video_path=args.video,
        image_path=args.image,
        audio_path=args.audio,
        steps=args.steps,
        seed=args.seed,
        device=args.cpu_device,
    )
    elapsed_seconds = time.perf_counter() - started_at
    summary = _summarize_audio_file(output_path)
    summary["elapsed_seconds"] = elapsed_seconds
    return summary


def _run_tt_reference(args: argparse.Namespace, tt_output: Path) -> dict:
    import ttnn

    from models.experimental.audiox.demo.tt_demo import run_tt_demo

    device = ttnn.open_device(device_id=args.tt_device_id)
    try:
        started_at = time.perf_counter()
        output_path = run_tt_demo(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            output=tt_output,
            device=device,
            video_path=args.video,
            image_path=args.image,
            audio_path=args.audio,
            steps=args.steps,
            seed=args.seed,
        )
        elapsed_seconds = time.perf_counter() - started_at
    finally:
        ttnn.close_device(device)

    summary = _summarize_audio_file(output_path)
    summary["elapsed_seconds"] = elapsed_seconds
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AudioX Stage 1 validation runner")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX .safetensors")
    p.add_argument("--prompt", type=str, default="", help="Optional text conditioning prompt")
    p.add_argument("--output-dir", type=Path, default=Path("audiox_validation"))
    visual = p.add_mutually_exclusive_group()
    visual.add_argument("--video", type=Path, help="Optional video prompt")
    visual.add_argument("--image", type=Path, help="Optional image prompt")
    p.add_argument("--audio", type=Path, help="Optional audio prompt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--tt", action="store_true", help="Also run the TT path and compare with CPU output")
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--report-json", type=Path, help="Optional explicit report path")
    args = p.parse_args(argv)
    if not args.prompt and args.video is None and args.image is None and args.audio is None:
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_output, tt_output, default_report_path = _build_output_paths(args.output_dir)
    report_path = args.report_json or default_report_path

    report = {
        "conditioning_mode": _infer_conditioning_mode(
            args.prompt,
            video_path=args.video,
            image_path=args.image,
            audio_path=args.audio,
        ),
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "seed": args.seed,
        "cpu": _run_cpu_reference(args, cpu_output),
        "tt": None,
        "comparison": None,
    }

    if args.tt:
        report["tt"] = _run_tt_reference(args, tt_output)
        report["comparison"] = _compare_audio_files(cpu_output, tt_output)

    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"conditioning_mode: {report['conditioning_mode']}")
    print(f"cpu_output: {cpu_output}")
    print(f"cpu_elapsed_seconds: {report['cpu']['elapsed_seconds']:.3f}")
    print(f"cpu_valid_16khz: {report['cpu']['valid_16khz']}")
    if report["tt"] is not None:
        print(f"tt_output: {tt_output}")
        print(f"tt_elapsed_seconds: {report['tt']['elapsed_seconds']:.3f}")
        print(f"tt_valid_16khz: {report['tt']['valid_16khz']}")
        print(f"tt_same_shape: {report['comparison']['same_shape']}")
        print(f"tt_same_sample_rate: {report['comparison']['same_sample_rate']}")
    print(f"report_json: {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
