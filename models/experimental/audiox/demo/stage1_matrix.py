# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path

from models.experimental.audiox.demo import validate as validate_mod


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a compact Stage 1 AudioX validation matrix")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX checkpoint")
    p.add_argument("--output-dir", type=Path, default=Path("audiox_stage1_matrix"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tt", action="store_true", help="Run TT alongside CPU")
    p.add_argument("--tt-device-id", type=int, default=0)
    return p.parse_args(argv)


def _build_cases() -> list[dict]:
    return [
        {
            "name": "text_to_audio",
            "prompt": "rain",
            "mode_label": "text-to-audio",
            "extra_args": [],
        },
        {
            "name": "text_to_music",
            "prompt": "piano music",
            "mode_label": "text-to-music",
            "extra_args": [],
        },
        {
            "name": "video_to_audio",
            "prompt": "",
            "mode_label": "video-to-audio",
            "extra_args": ["--synthetic-video"],
        },
        {
            "name": "video_to_music",
            "prompt": "music",
            "mode_label": "video-to-music",
            "extra_args": ["--synthetic-video"],
        },
    ]


def _build_case_argv(args: argparse.Namespace, case: dict, case_dir: Path) -> list[str]:
    argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(case_dir),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--mode-label",
        case["mode_label"],
    ]
    if case["prompt"]:
        argv.extend(["--prompt", case["prompt"]])
    if args.tt:
        argv.extend(["--tt", "--tt-device-id", str(args.tt_device_id)])
    argv.extend(case["extra_args"])
    return argv


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for case in _build_cases():
        case_dir = args.output_dir / case["name"]
        case_argv = _build_case_argv(args, case, case_dir)
        try:
            returncode = validate_mod.main(case_argv)
        except SystemExit as exc:
            returncode = int(exc.code) if isinstance(exc.code, int) else 1
        report_path = case_dir / "validation_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        results.append(
            {
                "name": case["name"],
                "returncode": returncode,
                "report_json": str(report_path),
                "stage1_checks": None if report is None else report.get("stage1_checks"),
            }
        )

        if returncode != 0:
            summary_path = args.output_dir / "matrix_summary.json"
            summary_path.write_text(json.dumps({"cases": results}, indent=2) + "\n")
            print(f"matrix_summary: {summary_path}")
            return returncode

    summary_path = args.output_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps({"cases": results}, indent=2) + "\n")
    print(f"matrix_summary: {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
