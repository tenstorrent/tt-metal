# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from pathlib import Path

from models.experimental.audiox.demo.demo import _HF_CONFIG, _resolve_duration_seconds
from models.experimental.audiox.demo.media import make_synthetic_video_prompt
from models.experimental.audiox.demo import validate as validate_mod


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Stage 3 multi-task switching matrix on a single TT session")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX checkpoint")
    p.add_argument("--output-dir", type=Path, default=Path("audiox_stage3_switch"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-seconds", type=int, default=_HF_CONFIG["duration_seconds"])
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--warmup-runs", type=int, default=1, help="Optional number of warmup runs before measuring")
    p.add_argument("--video-audio", type=Path, help="Optional real video for the video-to-audio case")
    p.add_argument("--video-music", type=Path, help="Optional real video for the video-to-music case")
    p.add_argument(
        "--synthetic-video",
        action="store_true",
        help="Use a deterministic in-memory video prompt for visual cases",
    )
    args = p.parse_args(argv)
    if args.synthetic_video and (args.video_audio is not None or args.video_music is not None):
        p.error("pass either --synthetic-video or real video paths, not both")
    return args


def _build_visual_cases(args: argparse.Namespace) -> tuple[Path | None, Path | None, bool]:
    if args.synthetic_video:
        return None, None, True
    if args.video_audio is None and args.video_music is None:
        return None, None, True
    return args.video_audio, (args.video_music or args.video_audio), False


def _build_cases(args: argparse.Namespace) -> list[dict]:
    video_audio_path, video_music_path, use_synthetic_video = _build_visual_cases(args)
    return [
        {
            "name": "text_to_audio",
            "prompt": "rain",
            "mode_label": "text-to-audio",
            "video_path": None,
            "video_prompt_tensor": None,
        },
        {
            "name": "text_to_music",
            "prompt": "piano music",
            "mode_label": "text-to-music",
            "video_path": None,
            "video_prompt_tensor": None,
        },
        {
            "name": "video_to_audio",
            "prompt": "",
            "mode_label": "video-to-audio",
            "video_path": video_audio_path,
            "video_prompt_tensor": use_synthetic_video,
        },
        {
            "name": "video_to_music",
            "prompt": "music",
            "mode_label": "video-to-music",
            "video_path": video_music_path,
            "video_prompt_tensor": use_synthetic_video,
        },
    ]


def _build_synthetic_video_prompt(args: argparse.Namespace):
    duration_seconds = _resolve_duration_seconds(args.duration_seconds)
    return make_synthetic_video_prompt(target_frames=5 * duration_seconds, image_size=224, seed=args.seed)


def _open_tt_device(device_id: int):
    from models.experimental.audiox.demo.tt_demo import open_tt_device

    return open_tt_device(device_id=device_id)


def _close_tt_device(device) -> None:
    from models.experimental.audiox.demo.tt_demo import close_tt_device

    close_tt_device(device)


def _build_session(checkpoint: Path, device, seed: int):
    from models.experimental.audiox.demo.tt_demo import TtAudioXSession

    return TtAudioXSession(checkpoint, device, seed=seed)


def _summarize_case(case: dict, details: dict, elapsed_seconds: float) -> dict:
    details["steps"] = case["steps"]
    summary = validate_mod._summarize_run_details(details["output_path"], elapsed_seconds, details)
    summary["name"] = case["name"]
    summary["conditioning_mode"] = case["mode_label"]
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args)
    synthetic_video_prompt = _build_synthetic_video_prompt(args) if _build_visual_cases(args)[2] else None

    warmup_reports = []
    case_reports = []
    setup_started_at = time.perf_counter()
    device = _open_tt_device(device_id=args.tt_device_id)
    session = _build_session(args.checkpoint, device, seed=args.seed)
    session_setup_seconds = time.perf_counter() - setup_started_at
    measured_started_at = time.perf_counter()

    try:
        for warmup_index in range(args.warmup_runs):
            warmup_case = cases[warmup_index % len(cases)]
            warmup_output = args.output_dir / f"warmup_run{warmup_index + 1}.wav"
            started_at = time.perf_counter()
            details = session.run(
                prompt=warmup_case["prompt"],
                output=warmup_output,
                video_path=warmup_case["video_path"],
                video_prompt_tensor=synthetic_video_prompt if warmup_case["video_prompt_tensor"] else None,
                steps=args.steps,
                seed=args.seed,
                duration_seconds=args.duration_seconds,
                return_details=True,
            )
            warmup_reports.append(_summarize_case({**warmup_case, "steps": args.steps}, details, time.perf_counter() - started_at))

        for case in cases:
            case_output = args.output_dir / case["name"] / "tt_output.wav"
            started_at = time.perf_counter()
            details = session.run(
                prompt=case["prompt"],
                output=case_output,
                video_path=case["video_path"],
                video_prompt_tensor=synthetic_video_prompt if case["video_prompt_tensor"] else None,
                steps=args.steps,
                seed=args.seed,
                duration_seconds=args.duration_seconds,
                return_details=True,
            )
            case_reports.append(_summarize_case({**case, "steps": args.steps}, details, time.perf_counter() - started_at))
    finally:
        session.deallocate()
        _close_tt_device(device)

    measured_elapsed_seconds = time.perf_counter() - measured_started_at
    summary = {
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "seed": args.seed,
        "duration_seconds": int(args.duration_seconds),
        "session_setup_seconds": session_setup_seconds,
        "measured_elapsed_seconds": measured_elapsed_seconds,
        "warmup_runs": warmup_reports,
        "cases": case_reports,
        "stage3_checks": {
            "all_required_modes_present": len(case_reports) == 4,
            "all_valid_16khz": all(case["valid_16khz"] for case in case_reports),
            "all_tt_runs_without_error": True,
        },
    }
    summary_path = args.output_dir / "stage3_switch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"stage3_switch_summary: {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
