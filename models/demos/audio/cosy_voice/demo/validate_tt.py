from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.audio.cosy_voice.demo.common import load_cases, runtime_summary, write_json
from models.demos.audio.cosy_voice.tt.pipeline import CosyVoicePipeline

DEMO_ROOT = Path(__file__).resolve().parent
ACCURACY_MANIFEST = DEMO_ROOT / "accuracy_cases.json"
PERFORMANCE_MANIFEST = DEMO_ROOT / "performance_cases.json"
QUALITY_MANIFEST = DEMO_ROOT / "quality_cases.json"


def validate_accuracy_metrics(token_accuracy_pct: float, minimum: float = 95.0) -> None:
    if token_accuracy_pct < minimum:
        raise AssertionError(f"Semantic token accuracy {token_accuracy_pct:.2f}% is below {minimum:.2f}%")


def validate_audio_metrics(sample_rate: int, audio_seconds: float, minimum_audio_seconds: float = 0.1) -> None:
    if sample_rate <= 0:
        raise AssertionError(f"Invalid sample rate: {sample_rate}")
    if audio_seconds < minimum_audio_seconds:
        raise AssertionError(f"Audio duration {audio_seconds:.4f}s is below {minimum_audio_seconds:.4f}s")


def validate_performance_metrics(
    tokens_per_second: float, rtf: float, min_tokens_per_second: float = 30.0, max_rtf: float = 0.5
) -> None:
    if tokens_per_second < min_tokens_per_second:
        raise AssertionError(
            f"Semantic generation speed {tokens_per_second:.2f} tok/s is below {min_tokens_per_second:.2f} tok/s"
        )
    if rtf >= max_rtf:
        raise AssertionError(f"RTF {rtf:.4f} is not below {max_rtf:.4f}")


def validate_quality_metrics(
    wer_pct: float, speaker_similarity_pct: float, max_wer_pct: float = 3.0, min_similarity_pct: float = 60.0
) -> None:
    if wer_pct >= max_wer_pct:
        raise AssertionError(f"WER {wer_pct:.4f}% is not below {max_wer_pct:.4f}%")
    if speaker_similarity_pct <= min_similarity_pct:
        raise AssertionError(f"Speaker similarity {speaker_similarity_pct:.4f}% is not above {min_similarity_pct:.4f}%")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Public CosyVoice accuracy/performance/quality runner")
    parser.add_argument("--suite", choices=("accuracy", "performance", "quality"), required=True)
    parser.add_argument("--reference-repo", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", default="/tmp/cosy_voice_outputs")
    parser.add_argument("--disable-text-frontend", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def manifest_for_suite(suite: str) -> Path:
    if suite == "accuracy":
        return ACCURACY_MANIFEST
    if suite == "performance":
        return PERFORMANCE_MANIFEST
    return QUALITY_MANIFEST


def run_suite(args: argparse.Namespace) -> dict:
    pipeline = CosyVoicePipeline(
        reference_repo=args.reference_repo,
        model_root=args.model_root,
        text_frontend=not args.disable_text_frontend,
    )
    try:
        cases = load_cases(manifest_for_suite(args.suite))
        report_cases = []
        for case in cases:
            output_path = str(Path(args.output_dir) / f"{case.name}.wav")
            prepared = pipeline.prepare_case(case)
            semantic_result = pipeline.generate_semantic_tokens(case, prepared=prepared)
            result = pipeline.synthesize_semantic_tokens(
                case, semantic_result, prepared=prepared, output_path=output_path
            )
            total_wall_seconds = semantic_result.wall_seconds + result.wall_seconds
            semantic_accuracy_pct = (
                pipeline.evaluate_semantic_token_accuracy(case, prepared=prepared) if args.suite == "accuracy" else None
            )
            quality_metrics = pipeline.evaluate_quality(case, output_path) if args.suite == "quality" else None
            entry = {
                "name": case.name,
                "mode": case.mode,
                "language": case.language,
                "model_dir": result.model_dir,
                "output_path": output_path,
                "sample_rate": result.sample_rate,
                "audio_seconds": result.audio_seconds,
                "wall_seconds": total_wall_seconds,
                "semantic_wall_seconds": semantic_result.wall_seconds,
                "acoustic_wall_seconds": result.wall_seconds,
                "rtf": total_wall_seconds / result.audio_seconds if result.audio_seconds > 0 else None,
                "semantic_token_count": semantic_result.token_count,
                "target_tokens_per_second": case.target_tokens_per_second,
                "target_rtf": case.target_rtf,
                "semantic_token_accuracy_pct": semantic_accuracy_pct,
                "semantic_tokens_per_second": semantic_result.tokens_per_second,
                "reference_text": None if quality_metrics is None else quality_metrics["reference_text"],
                "transcribed_text": None if quality_metrics is None else quality_metrics["transcribed_text"],
                "wer_pct": None if quality_metrics is None else quality_metrics["wer_pct"],
                "speaker_similarity_pct": None
                if quality_metrics is None
                else quality_metrics["speaker_similarity_pct"],
            }
            if args.check:
                validate_audio_metrics(entry["sample_rate"], entry["audio_seconds"])
            if args.check and args.suite == "performance":
                validate_performance_metrics(
                    entry["semantic_tokens_per_second"],
                    entry["rtf"],
                    min_tokens_per_second=case.target_tokens_per_second or 30.0,
                    max_rtf=case.target_rtf or 0.5,
                )
            if args.check and args.suite == "accuracy":
                validate_accuracy_metrics(entry["semantic_token_accuracy_pct"])
            if args.check and args.suite == "quality":
                validate_quality_metrics(entry["wer_pct"], entry["speaker_similarity_pct"])
            report_cases.append(entry)

        return {
            "suite": args.suite,
            "backend": "tt_semantic_tt_flow_frontend_length_regulator_torch_decoder_reference_vocoder",
            "semantic_backend": "tt_transformer",
            "acoustic_backend": "tt_flow_frontend_length_regulator_torch_decoder_reference_vocoder",
            "semantic_accuracy_mode": "teacher_forced_greedy_argmax" if args.suite == "accuracy" else None,
            "quality_asr_model": quality_metrics["quality_asr_model"]
            if args.suite == "quality" and report_cases
            else None,
            "throw_exception_on_fallback": bool(pipeline._ttnn.CONFIG.throw_exception_on_fallback)
            if pipeline._ttnn is not None
            else None,
            "runtime": runtime_summary(),
            "cases": report_cases,
        }
    finally:
        pipeline.close()


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report = run_suite(args)
    write_json(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
