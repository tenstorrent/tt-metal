# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from models.experimental.audiox.demo import stage1_matrix as matrix_mod


def test_build_case_argv_includes_tt_and_prompt_flags(tmp_path):
    args = matrix_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "4",
            "--seed",
            "7",
            "--tt",
            "--tt-device-id",
            "3",
        ]
    )
    case = {
        "name": "text_to_audio",
        "prompt": "rain",
        "mode_label": "text-to-audio",
        "extra_args": ["--synthetic-video"],
    }

    argv = matrix_mod._build_case_argv(args, case, tmp_path / case["name"])

    assert argv == [
        "--checkpoint",
        "/tmp/fake.safetensors",
        "--output-dir",
        str(tmp_path / case["name"]),
        "--steps",
        "4",
        "--seed",
        "7",
        "--mode-label",
        "text-to-audio",
        "--prompt",
        "rain",
        "--tt",
        "--tt-device-id",
        "3",
        "--synthetic-video",
    ]


def test_main_uses_validate_main_and_writes_summary(tmp_path, monkeypatch):
    reports = []

    def fake_validate_main(argv: list[str]) -> int:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_report.json").write_text('{"stage1_checks":{"valid_16khz":true}}\n')
        reports.append(argv)
        return 0

    monkeypatch.setattr(matrix_mod.validate_mod, "main", fake_validate_main)

    rc = matrix_mod.main(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "1",
        ]
    )

    assert rc == 0
    assert len(reports) == 4
    summary = (tmp_path / "matrix_summary.json").read_text()
    assert "text_to_audio" in summary
    assert "video_to_music" in summary
