from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.audio.cosy_voice.tt.pipeline import CosyVoicePipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Public CosyVoice demo runner")
    parser.add_argument("--mode", choices=("sft", "zero_shot", "cross_lingual", "instruct"), required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-repo", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--disable-text-frontend", action="store_true")
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.mode in {"sft", "instruct"} and not args.speaker_id:
        raise ValueError(f"--speaker-id is required for mode={args.mode}")
    if args.mode == "zero_shot" and (not args.prompt_text or not args.prompt_audio):
        raise ValueError("--prompt-text and --prompt-audio are required for zero_shot mode")
    if args.mode == "cross_lingual" and not args.prompt_audio:
        raise ValueError("--prompt-audio is required for cross_lingual mode")
    if args.mode == "instruct" and not args.instruction:
        raise ValueError("--instruction is required for instruct mode")


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_mode_args(args)

    pipeline = CosyVoicePipeline(
        reference_repo=args.reference_repo,
        model_root=args.model_root,
        text_frontend=not args.disable_text_frontend,
    )
    try:
        result = pipeline.generate_mode(
            mode=args.mode,
            text=args.text,
            output_path=args.output,
            speaker_id=args.speaker_id,
            prompt_text=args.prompt_text,
            prompt_audio=args.prompt_audio,
            instruction=args.instruction,
        )
        summary = {
            "mode": args.mode,
            "model_dir": result.model_dir,
            "output": args.output,
            "sample_rate": result.sample_rate,
            "audio_seconds": result.audio_seconds,
            "wall_seconds": result.wall_seconds,
            "rtf": result.rtf,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
