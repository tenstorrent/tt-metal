# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/utils/validate_perf_targets.py"


def _write_complete_run(path: Path, model: str, batch_size: int, seq_len: int, decode_tsu: float) -> None:
    payload = {
        "ml_model_name": model,
        "batch_size": batch_size,
        "input_sequence_length": seq_len,
        "measurements": [
            {"step_name": "inference_decode", "name": "tokens/s/user", "value": decode_tsu},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_validator(
    tmp_path: Path,
    strict_missing: bool = False,
    *,
    targets_yaml: Path | None = None,
    tests_yaml: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    benchmark_dir = tmp_path / "generated/benchmark_data"
    targets_yaml = targets_yaml or tmp_path / "models/model_targets.yaml"
    tests_yaml = tests_yaml or tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--targets-yaml",
        str(targets_yaml),
        "--benchmark-dir",
        str(benchmark_dir),
        "--tests-yaml",
        str(tests_yaml),
        "--sku",
        "wh_n150",
    ]
    if strict_missing:
        command.append("--strict-missing")
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=False,
    )


def test_validate_perf_targets_success(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=120.0,
    )

    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 1,
                                "seq_len": 128,
                                "status": "active",
                                "perf": {"decode_t/s/u": 100.0},
                                "accuracy": {},
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")

    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(
        yaml.safe_dump(tests_yaml), encoding="utf-8"
    )

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_perf_targets_detects_regression(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=40.0,
    )

    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 1,
                                "seq_len": 128,
                                "status": "active",
                                "perf": {"decode_t/s/u": 100.0},
                                "accuracy": {},
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(
        yaml.safe_dump(tests_yaml), encoding="utf-8"
    )

    result = _run_validator(tmp_path)
    assert result.returncode == 1
    assert "decode_t/s/u" in result.stdout


def test_validate_perf_targets_todo_entry_respects_strict_flag(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
    )

    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 1,
                                "seq_len": 128,
                                "status": "TODO",
                                "perf": {},
                                "accuracy": {},
                                "owner_id": "U000000",
                                "team": "models",
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(
        yaml.safe_dump(tests_yaml), encoding="utf-8"
    )

    non_strict = _run_validator(tmp_path, strict_missing=False)
    assert non_strict.returncode == 0, non_strict.stdout + non_strict.stderr

    strict = _run_validator(tmp_path, strict_missing=True)
    assert strict.returncode == 1


def test_validate_perf_targets_rejects_non_yaml_targets_path(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    invalid_targets = tmp_path / "models/model_targets.txt"
    invalid_targets.write_text("targets: {}", encoding="utf-8")
    tests_yaml = tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml"
    tests_yaml.write_text("[]", encoding="utf-8")

    result = _run_validator(tmp_path, targets_yaml=invalid_targets, tests_yaml=tests_yaml)
    assert result.returncode == 1
    assert "Invalid --targets-yaml" in result.stdout
