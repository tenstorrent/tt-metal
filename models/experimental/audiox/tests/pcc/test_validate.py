# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import torchaudio

from models.experimental.audiox.demo import validate as validate_mod


def test_infer_conditioning_mode_variants():
    assert validate_mod._infer_conditioning_mode("piano loop", video_path=None, image_path=None, audio_path=None) == "text-to-audio"
    assert (
        validate_mod._infer_conditioning_mode("crowd", video_path=Path("/tmp/in.mp4"), image_path=None, audio_path=None)
        == "visual+text-to-audio"
    )
    assert (
        validate_mod._infer_conditioning_mode("", video_path=Path("/tmp/in.mp4"), image_path=None, audio_path=None)
        == "video-to-audio"
    )
    assert (
        validate_mod._infer_conditioning_mode("", video_path=None, image_path=Path("/tmp/in.png"), audio_path=None)
        == "image-to-audio"
    )
    assert (
        validate_mod._infer_conditioning_mode("soft pad", video_path=None, image_path=None, audio_path=Path("/tmp/in.wav"))
        == "audio+text-to-audio"
    )


def test_summarize_audio_file_reports_16khz(tmp_path):
    output = tmp_path / "out.wav"
    torchaudio.save(str(output), torch.zeros(2, 32), 16000)
    summary = validate_mod._summarize_audio_file(output)
    assert summary["path"] == str(output)
    assert summary["sample_rate"] == 16000
    assert summary["num_channels"] == 2
    assert summary["num_frames"] == 32
    assert summary["valid_16khz"] is True


def test_compare_audio_files_shape_mismatch(tmp_path):
    ref = tmp_path / "ref.wav"
    cand = tmp_path / "cand.wav"
    torchaudio.save(str(ref), torch.zeros(2, 32), 16000)
    torchaudio.save(str(cand), torch.zeros(2, 40), 16000)
    comparison = validate_mod._compare_audio_files(ref, cand)
    assert comparison["same_sample_rate"] is True
    assert comparison["same_shape"] is False
    assert comparison["mae"] is None


def test_build_output_paths_defaults_under_output_dir(tmp_path):
    cpu_path, tt_path, report_path = validate_mod._build_output_paths(tmp_path)
    assert cpu_path == tmp_path / "cpu_reference.wav"
    assert tt_path == tmp_path / "tt_output.wav"
    assert report_path == tmp_path / "validation_report.json"


def test_summarize_run_details_prefers_sampling_window_for_diffusion_tps(tmp_path):
    output = tmp_path / "out.wav"
    torchaudio.save(str(output), torch.zeros(2, 16000), 16000)
    summary = validate_mod._summarize_run_details(
        output,
        elapsed_seconds=20.0,
        details={
            "conditioning_tokens": 10,
            "t_latent": 216,
            "steps": 2,
            "timings": {"sampling_seconds": 4.0, "generation_seconds": 20.0},
        },
    )
    assert summary["generation_seconds"] == 20.0
    assert summary["sampling_seconds"] == 4.0
    assert summary["diffusion_tokens_per_second"] == pytest.approx(108.0)
    assert summary["meets_stage1_diffusion_tps_ge_20"] is True
    assert summary["meets_stage1_generation_time_lt_30s"] is True


def test_parse_args_requires_at_least_one_conditioner_input():
    with pytest.raises(SystemExit) as exc:
        validate_mod._parse_args(["--checkpoint", "/tmp/fake.safetensors"])
    assert exc.value.code == 2


def test_parse_args_accepts_tt_validation_flags():
    args = validate_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--output-dir",
            "/tmp/audiox-validation",
            "--tt",
            "--tt-device-id",
            "3",
            "--report-json",
            "/tmp/report.json",
        ]
    )
    assert args.tt is True
    assert args.tt_device_id == 3
    assert args.output_dir == Path("/tmp/audiox-validation")
    assert args.report_json == Path("/tmp/report.json")


def test_parse_args_accepts_tt_warm_runs():
    args = validate_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--tt",
            "--tt-warm-runs",
            "3",
        ]
    )
    assert args.tt is True
    assert args.tt_warm_runs == 3


def test_parse_args_accepts_duration_override():
    args = validate_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--duration-seconds",
            "30",
        ]
    )
    assert args.duration_seconds == 30


def test_parse_args_tt_only_requires_tt():
    with pytest.raises(SystemExit) as exc:
        validate_mod._parse_args(
            [
                "--checkpoint",
                "/tmp/fake.safetensors",
                "--prompt",
                "wind chimes",
                "--tt-only",
            ]
        )
    assert exc.value.code == 2


def test_parse_args_accepts_tt_only_with_tt():
    args = validate_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--tt",
            "--tt-only",
        ]
    )
    assert args.tt is True
    assert args.tt_only is True


def test_warm_tt_runs_use_last_warm_latent_for_stage_checks(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    cpu_output = tmp_path / "cpu_reference.wav"
    tt_output = tmp_path / "tt_output.wav"

    def fake_run_cpu_reference(_args, _output, *, synthetic_video_prompt):
        assert synthetic_video_prompt is None
        torchaudio.save(str(cpu_output), torch.zeros(2, 16000), 16000)
        return {
            "path": str(cpu_output),
            "sample_rate": 16000,
            "num_frames": 16000,
            "num_channels": 2,
            "duration_seconds": 1.0,
            "valid_16khz": True,
            "elapsed_seconds": 1.0,
            "generation_seconds": 1.0,
            "conditioning_tokens": 4,
            "latent_tokens": 6,
            "diffusion_token_steps": 12,
            "sampling_seconds": 0.5,
            "diffusion_tokens_per_second": 24.0,
            "_latent": torch.tensor([1.0, 2.0]),
        }

    def fake_run_tt_reference(_args, _output, *, synthetic_video_prompt):
        assert synthetic_video_prompt is None
        torchaudio.save(str(tt_output), torch.zeros(2, 16000), 16000)
        return {
            "path": str(tt_output),
            "sample_rate": 16000,
            "num_frames": 16000,
            "num_channels": 2,
            "duration_seconds": 1.0,
            "valid_16khz": True,
            "elapsed_seconds": 5.0,
            "generation_seconds": 40.0,
            "conditioning_tokens": 4,
            "latent_tokens": 6,
            "diffusion_token_steps": 12,
            "sampling_seconds": 1.0,
            "diffusion_tokens_per_second": 12.0,
            "warm_runs": [
                {
                    "run_index": 2,
                    "elapsed_seconds": 2.0,
                    "generation_seconds": 20.0,
                    "sampling_seconds": 0.4,
                    "diffusion_tokens_per_second": 30.0,
                }
            ],
            "latent_comparison_anchor": "tt_warm_run_2",
            "_latent": torch.tensor([1.0, 2.0]),
        }

    monkeypatch.setattr(validate_mod, "_run_cpu_reference", fake_run_cpu_reference)
    monkeypatch.setattr(validate_mod, "_run_tt_reference", fake_run_tt_reference)

    exit_code = validate_mod.main(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--tt",
            "--tt-warm-runs",
            "2",
            "--output-dir",
            str(tmp_path),
            "--report-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = validate_mod.json.loads(report_path.read_text())
    assert report["latent_comparison"]["pcc"] == pytest.approx(1.0)
    assert report["tt"]["latent_comparison_anchor"] == "tt_warm_run_2"
    assert report["stage1_checks"]["tt_generation_time_lt_30s"] is False
    assert report["stage1_checks"]["tt_diffusion_tps_ge_20"] is False
    assert report["stage1_checks"]["latent_pcc_ge_0p95"] is True


def test_tt_only_skips_cpu_reference(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    tt_output = tmp_path / "tt_output.wav"

    def fail_cpu_reference(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("cpu reference should be skipped in --tt-only mode")

    def fake_run_tt_reference(_args, _output, *, synthetic_video_prompt):
        assert synthetic_video_prompt is None
        torchaudio.save(str(tt_output), torch.zeros(2, 16000), 16000)
        return {
            "path": str(tt_output),
            "sample_rate": 16000,
            "num_frames": 16000,
            "num_channels": 2,
            "duration_seconds": 1.0,
            "valid_16khz": True,
            "elapsed_seconds": 5.0,
            "generation_seconds": 20.0,
            "conditioning_tokens": 4,
            "latent_tokens": 6,
            "diffusion_token_steps": 12,
            "sampling_seconds": 0.4,
            "diffusion_tokens_per_second": 30.0,
            "_latent": torch.tensor([1.0, 2.0]),
        }

    monkeypatch.setattr(validate_mod, "_run_cpu_reference", fail_cpu_reference)
    monkeypatch.setattr(validate_mod, "_run_tt_reference", fake_run_tt_reference)

    exit_code = validate_mod.main(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "wind chimes",
            "--tt",
            "--tt-only",
            "--output-dir",
            str(tmp_path),
            "--report-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = validate_mod.json.loads(report_path.read_text())
    assert report["cpu"] is None
    assert report["comparison"] is None
    assert report["latent_comparison"] is None
    assert report["stage1_checks"]["valid_16khz"] is True
    assert report["stage1_checks"]["tt_generation_time_lt_30s"] is True
    assert report["stage1_checks"]["tt_diffusion_tps_ge_20"] is True
