# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio

from models.experimental.audiox.demo.demo import _HF_CONFIG, run_demo
from models.experimental.audiox.demo.media import make_synthetic_video_prompt


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


def _safe_pcc(lhs: torch.Tensor, rhs: torch.Tensor) -> float | None:
    if lhs.numel() == 0 or rhs.numel() == 0:
        return None
    lhs = lhs.reshape(-1).float()
    rhs = rhs.reshape(-1).float()
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    lhs_norm = torch.linalg.vector_norm(lhs)
    rhs_norm = torch.linalg.vector_norm(rhs)
    if lhs_norm == 0 or rhs_norm == 0:
        return None
    return float(torch.dot(lhs, rhs).item() / (lhs_norm.item() * rhs_norm.item()))


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    if not same_shape:
        return {
            "same_shape": False,
            "mae": None,
            "max_abs": None,
            "pcc": None,
            "token_cosine_ge_0p95_fraction": None,
        }

    reference = reference.float()
    candidate = candidate.float()
    delta = (reference - candidate).abs()

    token_fraction = None
    if reference.ndim == 3:
        ref_tokens = reference.transpose(1, 2).reshape(-1, reference.shape[1])
        cand_tokens = candidate.transpose(1, 2).reshape(-1, candidate.shape[1])
        cosine = torch.nn.functional.cosine_similarity(ref_tokens, cand_tokens, dim=1)
        token_fraction = float((cosine >= 0.95).float().mean().item())

    return {
        "same_shape": True,
        "mae": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
        "pcc": _safe_pcc(reference, candidate),
        "token_cosine_ge_0p95_fraction": token_fraction,
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
        "pcc": _safe_pcc(ref_audio, cand_audio),
    }


def _build_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "cpu_reference.wav",
        output_dir / "tt_output.wav",
        output_dir / "validation_report.json",
    )


def _build_synthetic_video_prompt(args: argparse.Namespace) -> torch.Tensor | None:
    if not args.synthetic_video:
        return None
    return make_synthetic_video_prompt(
        target_frames=5 * _HF_CONFIG["duration_seconds"],
        image_size=224,
        seed=args.seed,
    )


def _summarize_run_details(output_path: Path, elapsed_seconds: float, details: dict) -> dict:
    summary = _summarize_audio_file(output_path)
    summary["elapsed_seconds"] = elapsed_seconds
    timings = details.get("timings") or {}
    generation_seconds = timings.get("generation_seconds", elapsed_seconds)
    summary["generation_seconds"] = generation_seconds
    summary["conditioning_tokens"] = details["conditioning_tokens"]
    summary["latent_tokens"] = details["t_latent"]
    diffusion_token_steps = details["t_latent"] * details.get("steps", 0)
    summary["diffusion_token_steps"] = diffusion_token_steps
    summary["sampling_seconds"] = timings.get("sampling_seconds")
    throughput_window_seconds = summary["sampling_seconds"] or elapsed_seconds
    summary["diffusion_tokens_per_second"] = (
        0.0 if throughput_window_seconds == 0 else diffusion_token_steps / throughput_window_seconds
    )
    summary["meets_stage1_generation_time_lt_30s"] = generation_seconds < 30.0
    summary["meets_stage1_diffusion_tps_ge_20"] = summary["diffusion_tokens_per_second"] >= 20.0
    if timings:
        summary["timings"] = timings
    return summary


def _run_cpu_reference(args: argparse.Namespace, cpu_output: Path, *, synthetic_video_prompt: torch.Tensor | None) -> dict:
    started_at = time.perf_counter()
    details = run_demo(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        output=cpu_output,
        video_path=args.video,
        image_path=args.image,
        audio_path=args.audio,
        video_prompt_tensor=synthetic_video_prompt,
        steps=args.steps,
        seed=args.seed,
        device=args.cpu_device,
        return_details=True,
    )
    elapsed_seconds = time.perf_counter() - started_at
    details["steps"] = args.steps
    summary = _summarize_run_details(details["output_path"], elapsed_seconds, details)
    summary["latent_comparison_anchor"] = "cpu"
    summary["_latent"] = details["latent"]
    return summary


def _run_tt_reference(args: argparse.Namespace, tt_output: Path, *, synthetic_video_prompt: torch.Tensor | None) -> dict:
    from models.experimental.audiox.demo.tt_demo import TtAudioXSession, close_tt_device, open_tt_device, run_tt_demo

    if args.tt_warm_runs <= 1:
        device = open_tt_device(device_id=args.tt_device_id)
        try:
            started_at = time.perf_counter()
            details = run_tt_demo(
                checkpoint=args.checkpoint,
                prompt=args.prompt,
                output=tt_output,
                device=device,
                video_path=args.video,
                image_path=args.image,
                audio_path=args.audio,
                video_prompt_tensor=synthetic_video_prompt,
                steps=args.steps,
                seed=args.seed,
                return_details=True,
            )
            elapsed_seconds = time.perf_counter() - started_at
        finally:
            close_tt_device(device)

        details["steps"] = args.steps
        summary = _summarize_run_details(details["output_path"], elapsed_seconds, details)
        summary["_latent"] = details["latent"]
        return summary

    device = open_tt_device(device_id=args.tt_device_id)
    try:
        session = TtAudioXSession(args.checkpoint, device, seed=args.seed)
        warm_runs = []
        details = None
        elapsed_seconds = None
        for run_index in range(1, args.tt_warm_runs + 1):
            if run_index > 1:
                torch.manual_seed(args.seed)
            output_path = tt_output if run_index == 1 else tt_output.with_name(f"tt_output_run{run_index}.wav")
            started_at = time.perf_counter()
            run_details = session.run(
                prompt=args.prompt,
                output=output_path,
                video_path=args.video,
                image_path=args.image,
                audio_path=args.audio,
                video_prompt_tensor=synthetic_video_prompt,
                steps=args.steps,
                seed=args.seed,
                return_details=True,
            )
            run_elapsed_seconds = time.perf_counter() - started_at
            run_details["steps"] = args.steps
            run_summary = _summarize_run_details(run_details["output_path"], run_elapsed_seconds, run_details)
            if run_index == 1:
                details = run_details
                elapsed_seconds = run_elapsed_seconds
            else:
                warm_runs.append(
                    {
                        "run_index": run_index,
                        "elapsed_seconds": run_summary["elapsed_seconds"],
                        "generation_seconds": run_summary["generation_seconds"],
                        "sampling_seconds": run_summary["sampling_seconds"],
                        "diffusion_tokens_per_second": run_summary["diffusion_tokens_per_second"],
                        "decode_seconds": run_summary.get("timings", {}).get("decode_seconds"),
                        "save_seconds": run_summary.get("timings", {}).get("save_seconds"),
                    }
                )
    finally:
        close_tt_device(device)

    details["steps"] = args.steps
    summary = _summarize_run_details(details["output_path"], elapsed_seconds, details)
    summary["_latent"] = details["latent"]
    if warm_runs:
        summary["warm_runs"] = warm_runs
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
    p.add_argument("--synthetic-video", action="store_true", help="Use a deterministic in-memory video prompt")
    p.add_argument("--mode-label", type=str, help="Optional explicit conditioning mode label")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--tt", action="store_true", help="Also run the TT path and compare with CPU output")
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--tt-warm-runs", type=int, default=1, help="Optional number of additional warm TT runs")
    p.add_argument("--report-json", type=Path, help="Optional explicit report path")
    args = p.parse_args(argv)
    if args.synthetic_video and (args.video is not None or args.image is not None):
        p.error("pass at most one of --synthetic-video, --video, or --image")
    if not args.prompt and args.video is None and args.image is None and args.audio is None and not args.synthetic_video:
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_output, tt_output, default_report_path = _build_output_paths(args.output_dir)
    report_path = args.report_json or default_report_path
    synthetic_video_prompt = _build_synthetic_video_prompt(args)
    inferred_mode = _infer_conditioning_mode(
        args.prompt,
        video_path=args.video if not args.synthetic_video else Path("synthetic.mp4"),
        image_path=args.image,
        audio_path=args.audio,
    )

    report = {
        "conditioning_mode": args.mode_label or inferred_mode,
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "seed": args.seed,
        "cpu": _run_cpu_reference(args, cpu_output, synthetic_video_prompt=synthetic_video_prompt),
        "tt": None,
        "comparison": None,
        "latent_comparison": None,
        "stage1_checks": None,
    }

    if args.tt:
        gc.collect()
        report["tt"] = _run_tt_reference(args, tt_output, synthetic_video_prompt=synthetic_video_prompt)
        report["comparison"] = _compare_audio_files(cpu_output, tt_output)
        report["latent_comparison"] = _compare_tensors(report["cpu"]["_latent"], report["tt"]["_latent"])

    tt_perf_summary = None
    if report["tt"] is not None:
        warm_runs = report["tt"].get("warm_runs") or []
        if warm_runs:
            tt_perf_summary = warm_runs[-1]
        else:
            tt_perf_summary = report["tt"]

    report["stage1_checks"] = {
        "valid_16khz": bool(report["cpu"]["valid_16khz"]) and (
            report["tt"] is None or bool(report["tt"]["valid_16khz"])
        ),
        "tt_runs_without_error": report["tt"] is not None,
        "same_shape": None if report["comparison"] is None else report["comparison"]["same_shape"],
        "same_sample_rate": None if report["comparison"] is None else report["comparison"]["same_sample_rate"],
        "tt_generation_time_lt_30s": None
        if tt_perf_summary is None
        else tt_perf_summary["generation_seconds"] < 30.0,
        "tt_diffusion_tps_ge_20": None
        if tt_perf_summary is None
        else tt_perf_summary["diffusion_tokens_per_second"] >= 20.0,
        "latent_pcc_ge_0p95": None
        if report["latent_comparison"] is None or report["latent_comparison"]["pcc"] is None
        else report["latent_comparison"]["pcc"] >= 0.95,
    }

    report["cpu"].pop("_latent", None)
    if report["tt"] is not None:
        report["tt"].pop("_latent", None)

    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"conditioning_mode: {report['conditioning_mode']}")
    print(f"cpu_output: {cpu_output}")
    print(f"cpu_elapsed_seconds: {report['cpu']['elapsed_seconds']:.3f}")
    print(f"cpu_valid_16khz: {report['cpu']['valid_16khz']}")
    if report["tt"] is not None:
        print(f"tt_output: {tt_output}")
        print(f"tt_elapsed_seconds: {report['tt']['elapsed_seconds']:.3f}")
        print(f"tt_generation_seconds: {report['tt']['generation_seconds']:.3f}")
        print(f"tt_valid_16khz: {report['tt']['valid_16khz']}")
        print(f"tt_same_shape: {report['comparison']['same_shape']}")
        print(f"tt_same_sample_rate: {report['comparison']['same_sample_rate']}")
        print(f"tt_diffusion_tokens_per_second: {report['tt']['diffusion_tokens_per_second']:.3f}")
        if report["tt"].get("warm_runs"):
            warm_summary = report["tt"]["warm_runs"][-1]
            print(f"tt_warm_run_index: {warm_summary['run_index']}")
            print(f"tt_warm_generation_seconds: {warm_summary['generation_seconds']:.3f}")
            print(f"tt_warm_diffusion_tokens_per_second: {warm_summary['diffusion_tokens_per_second']:.3f}")
        if report["latent_comparison"] is not None:
            print(f"latent_pcc: {report['latent_comparison']['pcc']}")
    print(f"report_json: {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
