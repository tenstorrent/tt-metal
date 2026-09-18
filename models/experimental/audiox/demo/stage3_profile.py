# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path

from models.experimental.audiox.demo import validate as validate_mod
from models.experimental.audiox.demo.demo import _HF_CONFIG
from models.experimental.audiox.demo.tt_runtime import apply_tt_env_overrides, restore_tt_env

_TT_ENV_KEYS = (
    "AUDIOX_TT_OPEN_MODE",
    "AUDIOX_TT_LOCAL_MESH_WIDTH",
    "AUDIOX_TT_L1_SMALL_SIZE",
    "AUDIOX_TT_TRACE_REGION_SIZE",
    "AUDIOX_TT_NUM_COMMAND_QUEUES",
    "AUDIOX_TT_WORKER_L1_SIZE",
    "AUDIOX_TT_LONG_SEQUENCE_THRESHOLD",
    "AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK",
    "AUDIOX_TT_CONV1D_WIDTH_SLICES",
    "AUDIOX_TT_CONV1D_LOW_CHANNEL_WIDTH_SLICES",
    "AUDIOX_TT_CONV_TRANSPOSE_HEIGHT_SLICES",
    "AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD",
    "AUDIOX_TT_CONV_TRANSPOSE_LONG_HEIGHT_SLICES",
    "AUDIOX_TT_CONV_TRANSPOSE_LONG_WIDTH_SLICES",
    "AUDIOX_TT_CONV_TRANSPOSE_LONG_ACT_BLOCK_H",
    "AUDIOX_TT_CONV_TRANSPOSE_STRIDE2_ACT_BLOCK_H",
    "AUDIOX_TT_CONV_TRANSPOSE_STRIDE4_ACT_BLOCK_H",
    "AUDIOX_TT_OUT_CONV_STREAM_THRESHOLD",
    "AUDIOX_TT_RESIDUAL_STREAM_STRIDE4_THRESHOLD",
    "AUDIOX_TT_RESIDUAL_STREAM_STRIDE2_THRESHOLD",
    "AUDIOX_TT_CPU_DECODE",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Stage 3 AudioX validation pass with profiling artifacts")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX checkpoint")
    p.add_argument("--prompt", type=str, default="", help="Optional text conditioning prompt")
    p.add_argument("--output-dir", type=Path, default=Path("audiox_stage3_profile"))
    visual = p.add_mutually_exclusive_group()
    visual.add_argument("--video", type=Path, help="Optional video prompt")
    visual.add_argument("--image", type=Path, help="Optional image prompt")
    p.add_argument("--audio", type=Path, help="Optional audio prompt")
    p.add_argument("--synthetic-video", action="store_true", help="Use a deterministic in-memory video prompt")
    p.add_argument("--mode-label", type=str, help="Optional explicit conditioning mode label")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-seconds", type=int, default=_HF_CONFIG["duration_seconds"])
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--tt-warm-runs", type=int, default=2)
    p.add_argument("--tt-open-mode", choices=("mesh", "direct"))
    p.add_argument("--tt-local-mesh-width", type=int)
    p.add_argument("--tt-l1-small-size", type=int)
    p.add_argument("--tt-trace-region-size", type=int)
    p.add_argument("--tt-num-command-queues", type=int)
    p.add_argument("--tt-worker-l1-size", type=int)
    p.add_argument("--tt-long-sequence-threshold", type=int)
    p.add_argument("--tt-conv-transpose-input-chunk", type=int)
    p.add_argument("--tt-conv1d-width-slices", type=int)
    p.add_argument("--tt-conv1d-low-channel-width-slices", type=int)
    p.add_argument("--tt-conv-transpose-height-slices", type=int)
    p.add_argument("--tt-conv-transpose-long-threshold", type=int)
    p.add_argument("--tt-conv-transpose-long-height-slices", type=int)
    p.add_argument("--tt-conv-transpose-long-width-slices", type=int)
    p.add_argument("--tt-conv-transpose-long-act-block-h", type=int)
    p.add_argument("--tt-conv-transpose-stride2-act-block-h", type=int)
    p.add_argument("--tt-conv-transpose-stride4-act-block-h", type=int)
    p.add_argument("--tt-out-conv-stream-threshold", type=int)
    p.add_argument("--tt-residual-stream-stride4-threshold", type=int)
    p.add_argument("--tt-residual-stream-stride2-threshold", type=int)
    p.add_argument(
        "--rt-profiler-jsonl",
        type=Path,
        help="Optional path to write TT real-time profiler records as JSONL",
    )
    args = p.parse_args(argv)
    if args.synthetic_video and (args.video is not None or args.image is not None):
        p.error("pass at most one of --synthetic-video, --video, or --image")
    if (
        not args.prompt
        and args.video is None
        and args.image is None
        and args.audio is None
        and not args.synthetic_video
    ):
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def _build_validate_argv(args: argparse.Namespace, report_json: Path) -> list[str]:
    argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--report-json",
        str(report_json),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--duration-seconds",
        str(args.duration_seconds),
        "--tt",
        "--tt-only",
        "--tt-device-id",
        str(args.tt_device_id),
        "--tt-warm-runs",
        str(args.tt_warm_runs),
    ]
    if args.prompt:
        argv.extend(["--prompt", args.prompt])
    if args.video is not None:
        argv.extend(["--video", str(args.video)])
    if args.image is not None:
        argv.extend(["--image", str(args.image)])
    if args.audio is not None:
        argv.extend(["--audio", str(args.audio)])
    if args.synthetic_video:
        argv.append("--synthetic-video")
    if args.mode_label:
        argv.extend(["--mode-label", args.mode_label])
    return argv


def _record_to_row(record) -> dict:
    return {
        "program_id": record.program_id,
        "chip_id": record.chip_id,
        "start_timestamp": record.start_timestamp,
        "end_timestamp": record.end_timestamp,
        "frequency": record.frequency,
        "kernel_sources": list(record.kernel_sources),
    }


def _register_rt_profiler(jsonl_path: Path | None):
    if jsonl_path is None:
        return None, []

    try:
        import ttnn
    except Exception:
        return None, []

    register = getattr(getattr(ttnn, "device", None), "RegisterProgramRealtimeProfilerCallback", None)
    if register is None:
        return None, []

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out = jsonl_path.open("w", encoding="utf-8")
    lock = threading.Lock()
    rows = []

    def on_record(record):
        row = _record_to_row(record)
        with lock:
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()

    handle = register(on_record)
    return (ttnn, handle, out), rows


def _unregister_rt_profiler(registration) -> None:
    if registration is None:
        return
    ttnn, handle, out = registration
    try:
        unregister = getattr(getattr(ttnn, "device", None), "UnregisterProgramRealtimeProfilerCallback", None)
        if unregister is not None:
            unregister(handle)
    finally:
        out.close()


def _kernel_label(kernel_sources: list[str], program_id: int) -> str:
    if kernel_sources:
        return Path(kernel_sources[0]).name
    return f"program_{program_id}"


def _summarize_realtime_records(rows: list[dict]) -> dict:
    aggregates = defaultdict(lambda: {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0})
    total_device_seconds = 0.0

    for row in rows:
        duration_seconds = 0.0
        frequency = row.get("frequency") or 0
        start = row.get("start_timestamp") or 0
        end = row.get("end_timestamp") or 0
        if frequency > 0 and end >= start:
            duration_seconds = (end - start) / frequency

        label = _kernel_label(row.get("kernel_sources", []), row.get("program_id", -1))
        aggregates[label]["count"] += 1
        aggregates[label]["total_seconds"] += duration_seconds
        aggregates[label]["max_seconds"] = max(aggregates[label]["max_seconds"], duration_seconds)
        total_device_seconds += duration_seconds

    top_kernels = [
        {
            "kernel": kernel,
            "count": stats["count"],
            "total_seconds": stats["total_seconds"],
            "max_seconds": stats["max_seconds"],
        }
        for kernel, stats in sorted(aggregates.items(), key=lambda item: item[1]["total_seconds"], reverse=True)
    ]

    return {
        "record_count": len(rows),
        "kernel_count": len(top_kernels),
        "total_device_seconds": total_device_seconds,
        "top_kernels": top_kernels[:20],
    }


def _build_perf_summary(report: dict) -> dict:
    tt_summary = report["tt"]
    timings = tt_summary.get("timings", {})
    generation_seconds = tt_summary["generation_seconds"]
    decode_seconds = timings.get("decode_seconds")
    conditioning_seconds = timings.get("conditioning_seconds")
    sampling_seconds = tt_summary.get("sampling_seconds")
    decode_backend = timings.get("decode_backend")

    return {
        "model": "audiox",
        "conditioning_mode": report["conditioning_mode"],
        "duration_seconds": report["duration_seconds"],
        "steps": report["steps"],
        "generation_seconds": generation_seconds,
        "diffusion_tokens_per_second": tt_summary["diffusion_tokens_per_second"],
        "sampling_seconds": sampling_seconds,
        "decode_seconds": decode_seconds,
        "conditioning_seconds": conditioning_seconds,
        "decoder_share_of_generation": None
        if not decode_seconds or generation_seconds == 0
        else decode_seconds / generation_seconds,
        "stage3_checks": {
            "valid_16khz": bool(tt_summary["valid_16khz"]),
            "tt_runs_without_error": True,
            "diffusion_tps_ge_50": tt_summary["diffusion_tokens_per_second"] >= 50.0,
            "generation_time_lt_10s": generation_seconds < 10.0,
            "long_audio_ge_30s": report["duration_seconds"] >= 30,
            "decode_backend_is_tt": decode_backend == "tt",
        },
        "tt_timings": timings,
        "decode_backend": decode_backend,
        "decoder_profile": timings.get("decoder_profile"),
    }


def _tt_env_snapshot() -> dict:
    return {key: os.environ[key] for key in _TT_ENV_KEYS if key in os.environ}


def _apply_tt_env_overrides(args: argparse.Namespace) -> dict:
    return apply_tt_env_overrides(
        open_mode=args.tt_open_mode,
        local_mesh_width=args.tt_local_mesh_width,
        l1_small_size=args.tt_l1_small_size,
        trace_region_size=args.tt_trace_region_size,
        num_command_queues=args.tt_num_command_queues,
        worker_l1_size=args.tt_worker_l1_size,
        long_sequence_threshold=args.tt_long_sequence_threshold,
        conv_transpose_input_chunk=args.tt_conv_transpose_input_chunk,
        conv1d_width_slices=args.tt_conv1d_width_slices,
        conv1d_low_channel_width_slices=args.tt_conv1d_low_channel_width_slices,
        conv_transpose_height_slices=args.tt_conv_transpose_height_slices,
        conv_transpose_long_threshold=args.tt_conv_transpose_long_threshold,
        conv_transpose_long_height_slices=args.tt_conv_transpose_long_height_slices,
        conv_transpose_long_width_slices=args.tt_conv_transpose_long_width_slices,
        conv_transpose_long_act_block_h=args.tt_conv_transpose_long_act_block_h,
        conv_transpose_stride2_act_block_h=args.tt_conv_transpose_stride2_act_block_h,
        conv_transpose_stride4_act_block_h=args.tt_conv_transpose_stride4_act_block_h,
        out_conv_stream_threshold=args.tt_out_conv_stream_threshold,
        residual_stream_stride4_threshold=args.tt_residual_stream_stride4_threshold,
        residual_stream_stride2_threshold=args.tt_residual_stream_stride2_threshold,
        cpu_decode=False,
    )


def _restore_tt_env(previous: dict) -> None:
    restore_tt_env(previous)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "validation_report.json"
    summary_json = args.output_dir / "stage3_profile_summary.json"

    previous_tt_env = _apply_tt_env_overrides(args)
    registration, rows = _register_rt_profiler(args.rt_profiler_jsonl)
    exit_code = 1
    error_summary = None
    try:
        exit_code = validate_mod.main(_build_validate_argv(args, report_json))
    except Exception as exc:
        error_summary = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        _unregister_rt_profiler(registration)
        tt_env = _tt_env_snapshot()
        _restore_tt_env(previous_tt_env)

    report = json.loads(report_json.read_text()) if report_json.exists() else None
    perf_summary = None if report is None else _build_perf_summary(report)
    if (
        error_summary is None
        and exit_code == 0
        and perf_summary is not None
        and not perf_summary["stage3_checks"]["decode_backend_is_tt"]
    ):
        exit_code = 1
        error_summary = {
            "type": "Stage3ValidationError",
            "message": "Stage 3 validation requires decode_backend == 'tt'",
        }
    summary = {
        "report_json": str(report_json),
        "report_present": report is not None,
        "rt_profiler_jsonl": None if args.rt_profiler_jsonl is None else str(args.rt_profiler_jsonl),
        "tt_env": tt_env,
        "perf_summary": perf_summary,
        "realtime_profiler": _summarize_realtime_records(rows) if rows else None,
        "error": error_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"stage3_profile_summary: {summary_json}")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
