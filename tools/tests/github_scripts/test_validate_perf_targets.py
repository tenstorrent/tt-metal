# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/utils/validate_perf_targets.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_perf_targets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_run(
    path: Path,
    model: str,
    batch_size: int,
    seq_len: int,
    decode_tsu: float,
    extra_measurements: list[dict] | None = None,
) -> None:
    payload = {
        "ml_model_name": model,
        "batch_size": batch_size,
        "input_sequence_length": seq_len,
        "measurements": [
            {"step_name": "inference_decode", "name": "tokens/s/user", "value": decode_tsu},
        ],
    }
    if extra_measurements:
        payload["measurements"].extend(extra_measurements)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_validator(
    tmp_path: Path,
    strict_missing: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--path-profile",
        "cwd",
        "--sku",
        "wh_n150",
    ]
    if strict_missing:
        command.append("--strict-missing")
    return subprocess.run(
        command,
        cwd=str(tmp_path),
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
        decode_tsu=110.0,
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_perf_targets_supports_split_batch_perf_and_accuracy_entries(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    # Token-matching style artifact: batch=1 reports accuracy.
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=1024,
        decode_tsu=80.0,
        extra_measurements=[
            {"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 91.0},
        ],
    )

    # Eval-32 style artifact: batch=32 reports perf only (no top-k accuracy metrics).
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_2.json",
        model="demo-model",
        batch_size=32,
        seq_len=1024,
        decode_tsu=105.0,
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
                                "seq_len": 1024,
                                "status": "active",
                                "perf": {},
                                "accuracy": {"top1": 90.0},
                            },
                            {
                                "batch_size": 32,
                                "seq_len": 1024,
                                "status": "active",
                                "perf": {"decode_t/s/u": 100.0},
                                "accuracy": {},
                            },
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")

    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_gap_coverage_accepts_concrete_dims_only_entry(tmp_path):
    validator = _load_validator_module()
    tests_yaml_path = tmp_path / "models_e2e_tests.yaml"
    tests_yaml = [{"model": "llama3.2-1b", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    tests_yaml_path.write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    targets_yaml = {
        "version": 1,
        "targets": {
            "llama3.2-1b": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 32,
                                "seq_len": 1024,
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

    assert validator._validate_gap_coverage(tests_yaml_path, targets_yaml) == []


def test_validate_perf_targets_supports_compile_and_prefill_decode_metrics(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
        extra_measurements=[
            {"step_name": "compile_prefill", "name": "time(s)", "value": 15.0},
            {"step_name": "compile_decode", "name": "time(s)", "value": 8.0},
            {"step_name": "inference_prefill_decode", "name": "tokens/s/user", "value": 55.0},
        ],
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
                                "perf": {
                                    "compile_prefill": 15.0,
                                    "compile_decode": 8.0,
                                    "prefill_decode_t/s/u": 55.0,
                                },
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_perf_targets_supports_new_metric_names_and_ttft_ms_targets(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=4,
        seq_len=128,
        decode_tsu=33.0,
        extra_measurements=[
            {"step_name": "inference_prefill", "name": "time_to_token", "value": 0.11},
            {"step_name": "inference_decode", "name": "tokens/s", "value": 140.0},
        ],
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
                                "batch_size": 4,
                                "seq_len": 128,
                                "status": "active",
                                "perf": {
                                    "prefill_time_to_first_token": 120.0,
                                    "prefill_tolerance": 1.15,
                                    "decode_t/s/u": 30.0,
                                    "decode_t/s": 130.0,
                                    "decode_tolerance": 1.2,
                                },
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 1
    assert "decode_t/s/u" in result.stdout


def test_validate_perf_targets_local_artifacts_no_regression_then_regression(tmp_path):
    """
    End-to-end local validation test that does not require CI or model weights.

    The test creates complete_run_*.json artifacts under generated/benchmark_data
    and validates both a passing run and a failing run with a synthetic regression.
    """
    benchmark_dir = tmp_path / "generated/benchmark_data"
    benchmark_dir.mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    targets = {
        "version": 1,
        "targets": {
            "demo-model-a": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 1,
                                "seq_len": 128,
                                "status": "active",
                                "perf": {
                                    "decode_t/s/u": 100.0,
                                    "prefill_time_to_first_token": 0.10,
                                },
                                "accuracy": {
                                    "top1": 90.0,
                                },
                            }
                        ]
                    }
                },
            },
            "demo-model-b": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 4,
                                "seq_len": 256,
                                "status": "active",
                                "perf": {"decode_t/s/u": 50.0},
                                "accuracy": {},
                            }
                        ]
                    }
                },
            },
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")

    tests_yaml = [
        {"model": "demo-model-a", "skus": {"wh_n150": {"tier": 1}}, "team": "models"},
        {"model": "demo-model-b", "skus": {"wh_n150": {"tier": 2}}, "team": "models"},
    ]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    # Baseline artifacts: all metrics satisfy targets.
    (benchmark_dir / "complete_run_1.json").write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model-a",
                "batch_size": 1,
                "input_sequence_length": 128,
                "device_info": {"card_type": "wh_n150"},
                "measurements": [
                    {"step_name": "inference_decode", "name": "tokens/s/user", "value": 110.0},
                    {"step_name": "inference_prefill", "name": "time_to_token", "value": 0.09},
                    {"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 93.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (benchmark_dir / "complete_run_2.json").write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model-b",
                "batch_size": 4,
                "input_sequence_length": 256,
                "device_info": {"card_type": "wh_n150"},
                "measurements": [
                    {"step_name": "inference_decode", "name": "tokens/s/user", "value": 56.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    baseline = _run_validator(tmp_path)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr

    # Inject a regression for model-a:
    # - throughput drops below target
    # - lower-is-better prefill_time_to_first_token gets worse
    # - accuracy drops below target
    (benchmark_dir / "complete_run_1.json").write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model-a",
                "batch_size": 1,
                "input_sequence_length": 128,
                "device_info": {"card_type": "wh_n150"},
                "measurements": [
                    {"step_name": "inference_decode", "name": "tokens/s/user", "value": 40.0},
                    {"step_name": "inference_prefill", "name": "time_to_token", "value": 0.20},
                    {"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 70.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    regressed = _run_validator(tmp_path)
    assert regressed.returncode == 1
    assert "decode_t/s/u" in regressed.stdout
    assert "prefill_time_to_first_token" in regressed.stdout
    assert "top1" in regressed.stdout


def test_validate_perf_targets_respects_decode_tolerance_family(tmp_path):
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
                                "perf": {"decode_t/s/u": 100.0, "decode_tolerance": 1.25},
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_perf_targets_detects_accuracy_regression(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
        extra_measurements=[
            {"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 80.0},
        ],
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
                                "perf": {"decode_t/s/u": 90.0},
                                "accuracy": {"top1": 90.0},
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 1
    assert "top1" in result.stdout


def test_validate_perf_targets_requires_ttft_measurement_when_target_exists(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=4096,
        decode_tsu=100.0,
        # Intentionally omit inference_prefill.time_to_token.
        extra_measurements=[],
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
                                "seq_len": 4096,
                                "status": "active",
                                "perf": {"decode_t/s/u": 90.0, "prefill_time_to_first_token": 0.12},
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 1
    assert "prefill_time_to_first_token" in result.stdout


def test_validate_perf_targets_uses_decode_tps_target_without_recomputing_from_per_user_rate(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=4,
        seq_len=1024,
        decode_tsu=10.0,
        extra_measurements=[
            {"step_name": "inference_decode", "name": "tokens/s", "value": 123.0},
        ],
    )

    # decode_t/s intentionally does not equal decode_t/s/u * batch_size.
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 4,
                                "seq_len": 1024,
                                "status": "active",
                                "perf": {"decode_t/s/u": 10.0, "decode_t/s": 123.0},
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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


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
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    non_strict = _run_validator(tmp_path, strict_missing=False)
    assert non_strict.returncode == 0, non_strict.stdout + non_strict.stderr

    strict = _run_validator(tmp_path, strict_missing=True)
    assert strict.returncode == 1


def test_validate_perf_targets_rejects_unknown_path_profile(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--path-profile",
        "unknown-profile",
    ]
    result = subprocess.run(
        command,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=False,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_extract_metric_value_fails_for_ambiguous_unqualified_metric_name():
    validator = _load_validator_module()
    lookup = {
        ("inference_prefill", "token_verification"): 50.0,
        ("inference_decode", "token_verification"): 150.0,
    }

    try:
        validator._extract_metric_value("token_verification", lookup)
        assert False, "Expected ValueError for ambiguous metric lookup"
    except ValueError as exc:
        assert "ambiguous" in str(exc)
