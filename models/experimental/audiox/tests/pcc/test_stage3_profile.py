# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from models.experimental.audiox.demo import stage3_profile as stage3_profile_mod


def test_build_validate_argv_includes_tt_profile_path(tmp_path):
    args = stage3_profile_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/audiox.ckpt",
            "--prompt",
            "rain",
            "--output-dir",
            str(tmp_path),
            "--tt-warm-runs",
            "3",
        ]
    )
    argv = stage3_profile_mod._build_validate_argv(args, tmp_path / "report.json")
    assert "--tt" in argv
    assert "--tt-only" in argv
    assert "--tt-warm-runs" in argv
    assert str(tmp_path / "report.json") in argv


def test_tt_env_snapshot_and_restore(monkeypatch):
    monkeypatch.setenv("AUDIOX_TT_TRACE_REGION_SIZE", "123")
    args = stage3_profile_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/audiox.ckpt",
            "--prompt",
            "rain",
            "--tt-num-command-queues",
            "2",
            "--tt-long-sequence-threshold",
            "65536",
            "--tt-conv-transpose-input-chunk",
            "65536",
            "--tt-conv1d-width-slices",
            "4",
            "--tt-conv1d-low-channel-width-slices",
            "8",
            "--tt-conv-transpose-long-threshold",
            "55296",
            "--tt-conv-transpose-long-height-slices",
            "256",
            "--tt-conv-transpose-long-width-slices",
            "32",
            "--tt-conv-transpose-long-act-block-h",
            "32",
            "--tt-out-conv-stream-threshold",
            "1000000",
            "--tt-residual-stream-stride4-threshold",
            "999999999",
            "--tt-residual-stream-stride2-threshold",
            "220672",
        ]
    )
    previous = stage3_profile_mod._apply_tt_env_overrides(args)
    try:
        snapshot = stage3_profile_mod._tt_env_snapshot()
        assert snapshot["AUDIOX_TT_TRACE_REGION_SIZE"] == "123"
        assert snapshot["AUDIOX_TT_NUM_COMMAND_QUEUES"] == "2"
        assert snapshot["AUDIOX_TT_LONG_SEQUENCE_THRESHOLD"] == "65536"
        assert snapshot["AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK"] == "65536"
        assert snapshot["AUDIOX_TT_CONV1D_WIDTH_SLICES"] == "4"
        assert snapshot["AUDIOX_TT_CONV1D_LOW_CHANNEL_WIDTH_SLICES"] == "8"
        assert snapshot["AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD"] == "55296"
        assert snapshot["AUDIOX_TT_CONV_TRANSPOSE_LONG_HEIGHT_SLICES"] == "256"
        assert snapshot["AUDIOX_TT_CONV_TRANSPOSE_LONG_WIDTH_SLICES"] == "32"
        assert snapshot["AUDIOX_TT_CONV_TRANSPOSE_LONG_ACT_BLOCK_H"] == "32"
        assert snapshot["AUDIOX_TT_OUT_CONV_STREAM_THRESHOLD"] == "1000000"
        assert snapshot["AUDIOX_TT_RESIDUAL_STREAM_STRIDE4_THRESHOLD"] == "999999999"
        assert snapshot["AUDIOX_TT_RESIDUAL_STREAM_STRIDE2_THRESHOLD"] == "220672"
        assert snapshot["AUDIOX_TT_CPU_DECODE"] == "0"
    finally:
        stage3_profile_mod._restore_tt_env(previous)

    assert stage3_profile_mod.os.environ.get("AUDIOX_TT_TRACE_REGION_SIZE") == "123"
    assert "AUDIOX_TT_NUM_COMMAND_QUEUES" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_LONG_SEQUENCE_THRESHOLD" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV1D_WIDTH_SLICES" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV1D_LOW_CHANNEL_WIDTH_SLICES" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_LONG_HEIGHT_SLICES" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_LONG_WIDTH_SLICES" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_LONG_ACT_BLOCK_H" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_OUT_CONV_STREAM_THRESHOLD" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_RESIDUAL_STREAM_STRIDE4_THRESHOLD" not in stage3_profile_mod.os.environ
    assert "AUDIOX_TT_RESIDUAL_STREAM_STRIDE2_THRESHOLD" not in stage3_profile_mod.os.environ


def test_summarize_realtime_records_aggregates_kernel_durations():
    summary = stage3_profile_mod._summarize_realtime_records(
        [
            {
                "program_id": 1,
                "chip_id": 0,
                "start_timestamp": 10,
                "end_timestamp": 30,
                "frequency": 10,
                "kernel_sources": ["/tmp/a.cpp"],
            },
            {
                "program_id": 2,
                "chip_id": 0,
                "start_timestamp": 0,
                "end_timestamp": 50,
                "frequency": 10,
                "kernel_sources": ["/tmp/a.cpp"],
            },
            {
                "program_id": 3,
                "chip_id": 1,
                "start_timestamp": 0,
                "end_timestamp": 10,
                "frequency": 10,
                "kernel_sources": ["/tmp/b.cpp"],
            },
        ]
    )
    assert summary["record_count"] == 3
    assert summary["kernel_count"] == 2
    assert summary["top_kernels"][0]["kernel"] == "a.cpp"
    assert summary["top_kernels"][0]["count"] == 2
    assert summary["top_kernels"][0]["total_seconds"] == 7.0


def test_main_writes_stage3_profile_summary(tmp_path, monkeypatch):
    report_path = tmp_path / "validation_report.json"
    summary_path = tmp_path / "stage3_profile_summary.json"

    def fake_validate_main(argv):
        report_arg = Path(argv[argv.index("--report-json") + 1])
        report_arg.write_text(
            json.dumps(
                {
                    "conditioning_mode": "text-to-audio",
                    "duration_seconds": 10,
                    "steps": 1,
                    "tt": {
                        "valid_16khz": True,
                        "generation_seconds": 9.5,
                        "diffusion_tokens_per_second": 55.0,
                        "sampling_seconds": 1.25,
                        "timings": {
                            "decode_seconds": 7.0,
                            "conditioning_seconds": 0.5,
                            "decode_backend": "tt",
                            "decoder_profile": {
                                "blocks": [
                                    {
                                        "block_index": 4,
                                        "stride": 2,
                                        "out_channels": 128,
                                        "seconds": 31.5,
                                    }
                                ]
                            },
                        },
                    },
                }
            )
        )
        return 0

    monkeypatch.setattr(stage3_profile_mod.validate_mod, "main", fake_validate_main)
    monkeypatch.setattr(stage3_profile_mod, "_register_rt_profiler", lambda _path: ("reg", []))
    monkeypatch.setattr(stage3_profile_mod, "_unregister_rt_profiler", lambda _reg: None)

    exit_code = stage3_profile_mod.main(
        [
            "--checkpoint",
            "/tmp/audiox.ckpt",
            "--prompt",
            "rain",
            "--output-dir",
            str(tmp_path),
            "--tt-trace-region-size",
            "1024",
            "--tt-num-command-queues",
            "2",
            "--tt-conv-transpose-input-chunk",
            "65536",
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["report_json"] == str(report_path)
    assert summary["tt_env"]["AUDIOX_TT_TRACE_REGION_SIZE"] == "1024"
    assert summary["tt_env"]["AUDIOX_TT_NUM_COMMAND_QUEUES"] == "2"
    assert summary["tt_env"]["AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK"] == "65536"
    assert summary["tt_env"]["AUDIOX_TT_CPU_DECODE"] == "0"
    assert summary["perf_summary"]["generation_seconds"] == 9.5
    assert summary["perf_summary"]["decode_backend"] == "tt"
    assert summary["perf_summary"]["decoder_profile"]["blocks"][0]["seconds"] == 31.5
    assert summary["perf_summary"]["tt_timings"]["decode_seconds"] == 7.0
    assert summary["perf_summary"]["stage3_checks"]["diffusion_tps_ge_50"] is True
    assert summary["perf_summary"]["stage3_checks"]["generation_time_lt_10s"] is True
    assert summary["perf_summary"]["stage3_checks"]["decode_backend_is_tt"] is True


def test_main_rejects_cpu_decode_stage3_summary(tmp_path, monkeypatch):
    def fake_validate_main(argv):
        report_arg = Path(argv[argv.index("--report-json") + 1])
        report_arg.write_text(
            json.dumps(
                {
                    "conditioning_mode": "text-to-audio",
                    "duration_seconds": 10,
                    "steps": 1,
                    "tt": {
                        "valid_16khz": True,
                        "generation_seconds": 7.0,
                        "diffusion_tokens_per_second": 60.0,
                        "sampling_seconds": 1.0,
                        "timings": {
                            "decode_seconds": 5.0,
                            "conditioning_seconds": 0.5,
                            "decode_backend": "cpu",
                        },
                    },
                }
            )
        )
        return 0

    monkeypatch.setattr(stage3_profile_mod.validate_mod, "main", fake_validate_main)
    monkeypatch.setattr(stage3_profile_mod, "_register_rt_profiler", lambda _path: ("reg", []))
    monkeypatch.setattr(stage3_profile_mod, "_unregister_rt_profiler", lambda _reg: None)

    exit_code = stage3_profile_mod.main(
        [
            "--checkpoint",
            "/tmp/audiox.ckpt",
            "--prompt",
            "rain",
            "--output-dir",
            str(tmp_path),
        ]
    )

    summary = json.loads((tmp_path / "stage3_profile_summary.json").read_text())
    assert exit_code == 1
    assert summary["perf_summary"]["decode_backend"] == "cpu"
    assert summary["perf_summary"]["stage3_checks"]["decode_backend_is_tt"] is False
    assert summary["error"] == {
        "type": "Stage3ValidationError",
        "message": "Stage 3 validation requires decode_backend == 'tt'",
    }


def test_main_writes_summary_when_validate_fails(tmp_path, monkeypatch):
    def fake_validate_main(_argv):
        raise RuntimeError("tt failed")

    monkeypatch.setattr(stage3_profile_mod.validate_mod, "main", fake_validate_main)
    monkeypatch.setattr(stage3_profile_mod, "_register_rt_profiler", lambda _path: ("reg", []))
    monkeypatch.setattr(stage3_profile_mod, "_unregister_rt_profiler", lambda _reg: None)

    exit_code = stage3_profile_mod.main(
        [
            "--checkpoint",
            "/tmp/audiox.ckpt",
            "--prompt",
            "rain",
            "--output-dir",
            str(tmp_path),
            "--tt-num-command-queues",
            "2",
        ]
    )

    summary = json.loads((tmp_path / "stage3_profile_summary.json").read_text())
    assert exit_code == 1
    assert summary["report_present"] is False
    assert summary["perf_summary"] is None
    assert summary["tt_env"]["AUDIOX_TT_NUM_COMMAND_QUEUES"] == "2"
    assert summary["error"] == {"type": "RuntimeError", "message": "tt failed"}
