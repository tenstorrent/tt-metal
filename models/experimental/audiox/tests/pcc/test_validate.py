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
            "timings": {"sampling_seconds": 4.0},
        },
    )
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
