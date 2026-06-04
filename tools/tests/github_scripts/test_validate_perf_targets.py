# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/utils/validate_perf_targets.py"
LLM_DEMO_UTILS_PATH = REPO_ROOT / "models/demos/utils/llm_demo_utils.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_perf_targets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_llm_demo_utils_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if "loguru" not in sys.modules:
        logger_stub = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
        sys.modules["loguru"] = types.SimpleNamespace(logger=logger_stub)

    if "models.perf.benchmarking_utils" not in sys.modules:
        benchmarking_stub = types.SimpleNamespace(
            BenchmarkData=object,
            BenchmarkProfiler=object,
        )
        sys.modules["models.perf.benchmarking_utils"] = benchmarking_stub

    spec = importlib.util.spec_from_file_location("llm_demo_utils", LLM_DEMO_UTILS_PATH)
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
    extra_env: dict[str, str] | None = None,
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
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), **(extra_env or {})},
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


def test_validate_perf_targets_supports_per_metric_tolerance_override(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=118.0,
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
                                    "decode_t/s/u": 100.0,
                                    "decode_t_s_u_tolerance": 0.2,
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


def test_validate_perf_targets_rejects_tolerance_outside_fraction_range(tmp_path):
    validator = _load_validator_module()
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
                                "perf": {"decode_t/s/u": 100.0, "decode_t_s_u_tolerance": 1.2},
                                "accuracy": {},
                            }
                        ]
                    }
                },
            }
        },
    }

    errors = validator._validate_targets_schema(targets)
    assert any("outside [0.0, 1.0]" in err for err in errors)


def test_validate_perf_targets_supports_prefill_time_to_first_token_ms_units(tmp_path):
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
            {"step_name": "inference_prefill", "name": "time_to_token", "value": 0.08},
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
                                "perf": {"decode_t/s/u": 100.0, "prefill_time_to_first_token": 80.0},
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


def test_validate_perf_targets_prefers_prefill_time_to_first_token_over_prefill_time_to_token(tmp_path):
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
            {"step_name": "inference_prefill", "name": "time_to_token", "value": 0.08},
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
                                    "decode_t/s/u": 100.0,
                                    "prefill_time_to_token": 0.01,
                                    "prefill_time_to_first_token": 80.0,
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
    assert "ignoring prefill_time_to_token" in result.stdout


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
            {"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 70.0},
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


def test_verify_perf_prefers_prefill_time_to_first_token_with_unit_conversion(monkeypatch):
    llm_demo_utils = _load_llm_demo_utils_module()
    warnings_seen = []
    monkeypatch.setattr(llm_demo_utils.logger, "warning", lambda message: warnings_seen.append(message))

    llm_demo_utils.verify_perf(
        measurements={
            "prefill_time_to_token": 0.08,
            "prefill_time_to_first_token": 80.0,
        },
        expected_perf_metrics={
            "prefill_time_to_token": 0.05,
            "prefill_time_to_first_token": 80.0,
        },
        expected_measurements={
            "prefill_time_to_token": True,
            "prefill_time_to_first_token": True,
        },
    )

    warning_text = " ".join(warnings_seen)
    assert "ignoring prefill_time_to_token" in warning_text


def test_verify_perf_respects_metric_specific_tolerance(monkeypatch):
    llm_demo_utils = _load_llm_demo_utils_module()

    llm_demo_utils.verify_perf(
        measurements={
            "decode_t/s/u": 118.0,
        },
        expected_perf_metrics={
            "decode_t/s/u": 100.0,
            "decode_t_s_u_tolerance": 0.2,
        },
        expected_measurements={
            "decode_t/s/u": True,
        },
    )


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
