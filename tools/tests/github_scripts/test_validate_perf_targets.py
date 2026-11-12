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
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/utils/validate_perf_targets.py"
LLM_DEMO_UTILS_PATH = REPO_ROOT / "models/demos/utils/llm_demo_utils.py"
COMMON_DEMO_UTILS_PATH = REPO_ROOT / "models/demos/utils/common_demo_utils.py"


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


def _load_common_demo_utils_module():
    """Load common_demo_utils with its heavy top-level deps stubbed.

    report_generation_rate / report_e2e_metrics / report_vision_fps only need cv2, torch,
    datasets, etc. to exist as importable names; they never use them (the perf helpers import
    benchmarking_utils and device_sku lazily inside the function body). Stub whatever is not
    already imported so the lightweight github-scripts test env can load the module.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    def _ensure(name, **attrs):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            if not hasattr(mod, key):
                setattr(mod, key, value)
        return mod

    # common_demo_utils evaluates cv2.IMREAD_COLOR in a module-level default argument, so the
    # stub needs that attribute present at import time.
    _ensure("cv2", IMREAD_COLOR=1)
    _ensure("numpy")
    torch_stub = _ensure("torch")
    # common_demo_utils defines `class Ensemble(nn.ModuleList)` at module level, so the stub
    # must expose ModuleList as a usable base class.
    nn_stub = _ensure("torch.nn", ModuleList=type("ModuleList", (), {}))
    if not hasattr(torch_stub, "nn"):
        torch_stub.nn = nn_stub
    _ensure("torchvision", transforms=types.SimpleNamespace())
    _ensure("torchvision.transforms")
    _ensure("datasets", load_dataset=lambda *a, **k: None)
    _ensure("loguru", logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
    _ensure("PIL", Image=types.SimpleNamespace())
    _ensure("tqdm", tqdm=lambda iterable=None, *a, **k: iterable)
    _ensure("ttnn")

    spec = importlib.util.spec_from_file_location("common_demo_utils", COMMON_DEMO_UTILS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_benchmark_stub(monkeypatch, is_ci: bool, saved: list):
    """Stub the lazily-imported benchmarking_utils / device_sku modules used by the perf helpers.

    `saved` collects one dict per save_partial_run_json call so tests can assert the measurement
    pairs and run metadata the helper builds.
    """

    class _Profiler:
        def start(self, *a, **k):
            pass

        def end(self, *a, **k):
            pass

        def get_duration(self, *a, **k):
            return 0.0

    class _Data:
        def __init__(self):
            self.measurements = []

        def add_measurement(self, profiler, iteration, step_name, measurement_name, value, *a, **k):
            self.measurements.append((step_name, measurement_name, value))

        def save_partial_run_json(self, profiler, **kwargs):
            saved.append({"measurements": list(self.measurements), **kwargs})

    monkeypatch.setitem(
        sys.modules,
        "models.perf.benchmarking_utils",
        types.SimpleNamespace(IS_CI_ENV=is_ci, BenchmarkData=_Data, BenchmarkProfiler=_Profiler),
    )
    monkeypatch.setitem(
        sys.modules,
        "models.demos.utils.device_sku",
        types.SimpleNamespace(get_current_device_sku_name=lambda: "wh_n150"),
    )


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


def _write_vision_run(path: Path, model: str, batch_size: int, top1: float, fps: float) -> None:
    """A vision classifier run: accuracy and throughput measured in the same pass."""
    payload = {
        "ml_model_name": model,
        "batch_size": batch_size,
        "measurements": [
            {"step_name": "inference", "name": "top1_accuracy", "value": top1},
            {"step_name": "inference", "name": "fps", "value": fps},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_vision_targets(path: Path, model: str, batch_size: int, top1: float, fps: float) -> None:
    targets = {
        "version": 1,
        "targets": {
            model: {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": batch_size,
                                "status": "active",
                                "perf": {"fps": fps, "fps_tolerance": 0.15},
                                "accuracy": {"top1": top1, "top1_tolerance": 0.01},
                            }
                        ]
                    }
                },
            }
        },
    }
    path.write_text(yaml.safe_dump(targets), encoding="utf-8")


def _vision_scaffold(tmp_path: Path, model: str) -> None:
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)
    tests_yaml = [{"model": model, "skus": {"wh_n150": {"tier": 1}}, "team": "ttnn"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")


def test_validate_perf_targets_checks_vision_accuracy_and_throughput_in_one_run(tmp_path):
    """A single-pass vision run must have BOTH its accuracy and perf targets checked.

    LLM runs are split: token-matching runs only validate accuracy, eval runs only
    validate perf. Vision classifiers report both from one pass, so neither metric may
    be skipped.
    """
    _vision_scaffold(tmp_path, "resnet50")
    _write_vision_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="resnet50",
        batch_size=16,
        top1=75.375,
        fps=28.43,
    )
    _write_vision_targets(tmp_path / "models/model_targets.yaml", "resnet50", 16, 75.375, 28.43)

    result = _run_validator(tmp_path, strict_missing=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # Both metrics reported, neither silently skipped by the accuracy/perf split.
    assert "top1" in output and "fps" in output, output


def test_validate_perf_targets_detects_vision_throughput_regression(tmp_path):
    _vision_scaffold(tmp_path, "resnet50")
    _write_vision_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="resnet50",
        batch_size=16,
        top1=75.375,
        fps=10.0,  # far below the 15% tolerance band around 28.43
    )
    _write_vision_targets(tmp_path / "models/model_targets.yaml", "resnet50", 16, 75.375, 28.43)

    result = _run_validator(tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr


def test_validate_perf_targets_detects_vision_accuracy_regression(tmp_path):
    _vision_scaffold(tmp_path, "resnet50")
    _write_vision_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="resnet50",
        batch_size=16,
        top1=60.0,  # far below the 1% tolerance band around 75.375
        fps=28.43,
    )
    _write_vision_targets(tmp_path / "models/model_targets.yaml", "resnet50", 16, 75.375, 28.43)

    result = _run_validator(tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr


def test_llm_accuracy_run_still_skips_perf_targets(tmp_path):
    """Guard the vision change against leaking into LLM runs.

    A token-matching run reports top1_token_accuracy, which must NOT mark the run as
    single-pass, so its (teacher-forced, meaningless) throughput stays unchecked.
    """
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=1.0,  # would fail the perf target if it were checked
        extra_measurements=[{"step_name": "inference_decode", "name": "top1_token_accuracy", "value": 99.0}],
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
                                "accuracy": {"top1": 99.0},
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
    assert "100.0" in non_strict.stdout
    assert "no-target" in non_strict.stdout

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


def _write_single_target(tmp_path: Path, perf: dict, accuracy: dict | None = None) -> None:
    """Write a minimal targets + tests YAML pair for a demo-model/wh_n150 entry."""
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
                                "perf": perf,
                                "accuracy": accuracy or {},
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


def test_report_shows_measured_and_target_on_pass(tmp_path):
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
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0})

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    # Both the measured value and the target are surfaced in the report table.
    assert "decode_t/s/u" in result.stdout
    assert "110.0" in result.stdout
    assert "100.0" in result.stdout
    assert "pass" in result.stdout


def test_report_shows_measurement_without_target(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    # decode has a target; prefill_decode t/s/u is measured but has no target entry.
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
        extra_measurements=[
            {"step_name": "inference_prefill_decode", "name": "tokens/s/user", "value": 55.0},
        ],
    )
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0})

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no-target" in result.stdout
    assert "55.0" in result.stdout


def test_report_written_to_github_step_summary(tmp_path):
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
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0})

    summary_path = tmp_path / "summary.md"
    result = _run_validator(tmp_path, extra_env={"GITHUB_STEP_SUMMARY": str(summary_path)})
    assert result.returncode == 0, result.stdout + result.stderr

    summary = summary_path.read_text(encoding="utf-8")
    assert "Report and Validate Perf and Accuracy targets" in summary
    assert "decode_t/s/u" in summary
    assert "110.0" in summary


def test_report_present_on_regression(tmp_path):
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
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0})

    result = _run_validator(tmp_path)
    # Regression still fails the step, but the measured value is reported too.
    assert result.returncode == 1
    assert "40.0" in result.stdout
    assert "fail" in result.stdout


def test_render_summary_empty_reports_placeholder():
    validator = _load_validator_module()
    empty_result = validator.ValidationResult()
    summary = validator._render_summary(empty_result)
    assert "No perf/accuracy measurements reported" in summary


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


@pytest.mark.parametrize("metric", ["ifeval", "gpqa", "gsm8k"])
def test_task_accuracy_and_performance_from_the_same_run(tmp_path, metric):
    _vision_scaffold(tmp_path, "task-demo")
    accuracy_target, tolerance = (100, 0.2) if metric == "gpqa" else (80, 0.05)
    target = {
        "version": 1,
        "targets": {
            "task-demo": {
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "batch_size": 1,
                                "status": "active",
                                "accuracy": {metric: accuracy_target, f"{metric}_tolerance": tolerance},
                                "perf": {"decode_t/s/u": 120, "decode_t/s/u_tolerance": 0.15},
                            }
                        ]
                    }
                }
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(target))
    path = tmp_path / "generated/benchmark_data/complete_run_task.json"
    cases = [(80, 120, 0), (60, 120, 1), (80, 60, 1)]
    if metric == "gpqa":
        cases += [(90, 120, 0), (100, 120, 0), (79, 120, 1)]
    for accuracy, throughput, expected_code in cases:
        _write_complete_run(
            path,
            model="task-demo",
            batch_size=1,
            seq_len=128,
            decode_tsu=throughput,
            extra_measurements=[{"step_name": "inference", "name": f"{metric}_accuracy", "value": accuracy}],
        )
        result = _run_validator(tmp_path)
        assert result.returncode == expected_code, result.stdout + result.stderr
        assert metric in result.stdout and "decode_t/s/u" in result.stdout


@pytest.mark.parametrize("metric", ["text_image_pcc", "edit_image_pcc"])
@pytest.mark.parametrize("score,success", [(0.99, True), (0.96, False), (None, False)])
def test_image_agreement_targets_and_throughput(tmp_path, metric, score, success):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)
    _write_single_target(tmp_path, perf={"fps": 0.04}, accuracy={metric: 1.0, f"{metric}_tolerance": 0.03})
    measurements = [{"step_name": "inference", "name": "fps", "value": 0.04}]
    if score is not None:
        measurements.append({"step_name": "inference", "name": metric, "value": score})
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_image.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=0,
        extra_measurements=measurements,
    )
    result = _run_validator(tmp_path)
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert metric in result.stdout
    assert "fps" in result.stdout


@pytest.mark.parametrize("metric", ["ifeval", "gpqa", "gsm8k"])
def test_throughput_only_task_payload_fails_missing_accuracy(tmp_path, metric):
    _vision_scaffold(tmp_path, "demo-model")
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0}, accuracy={metric: 80})
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_missing_score.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
    )
    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"metric '{metric}' missing in benchmark payload" in result.stdout


@pytest.mark.parametrize("metric", ["gsm8k", "text_image_pcc", "edit_image_pcc"])
def test_accuracy_only_target_requires_score_without_perf_target(tmp_path, metric):
    _vision_scaffold(tmp_path, "demo-model")
    _write_single_target(tmp_path, perf={}, accuracy={metric: 1.0})
    path = tmp_path / "generated/benchmark_data/complete_run_accuracy_only.json"
    path.write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model",
                "batch_size": 1,
                "input_sequence_length": 128,
                "measurements": [{"step_name": "inference", "name": "fps", "value": 0.04}],
            }
        )
    )
    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"metric '{metric}' missing in benchmark payload" in result.stdout


def test_llm_perf_run_still_skips_token_accuracy_targets(tmp_path):
    _vision_scaffold(tmp_path, "demo-model")
    _write_single_target(tmp_path, perf={"decode_t/s/u": 100.0}, accuracy={"top1": 99.0, "top5": 99.0})
    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_perf.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=100.0,
    )
    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "decode_t/s/u" in result.stdout
    assert "missing-measurement" not in result.stdout


def test_throughput_only_classifier_payload_fails_missing_accuracy(tmp_path):
    _vision_scaffold(tmp_path, "demo-model")
    _write_single_target(tmp_path, perf={"fps": 30.0}, accuracy={"top1": 75.0})
    path = tmp_path / "generated/benchmark_data/complete_run_classifier.json"
    path.write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model",
                "batch_size": 1,
                "input_sequence_length": 128,
                "measurements": [{"step_name": "inference", "name": "fps", "value": 30.0}],
            }
        )
    )
    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "metric 'top1' missing in benchmark payload" in result.stdout


def test_report_shows_measurement_when_entry_missing(tmp_path):
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
    targets = {"version": 1, "targets": {}}
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 1}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "110.0" in result.stdout
    assert "no-target" in result.stdout
    assert "::error::" not in result.stdout


def test_enforce_perf_false_does_not_fail_on_perf_miss(tmp_path):
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_complete_run(
        tmp_path / "generated/benchmark_data/complete_run_1.json",
        model="demo-model",
        batch_size=1,
        seq_len=128,
        decode_tsu=10.0,
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
                                "enforce_perf": False,
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
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 3}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "report-only" in result.stdout
    assert "10.0" in result.stdout
    assert "100.0" in result.stdout
    assert "::error::" not in result.stdout


def test_enforce_perf_false_still_fails_accuracy(tmp_path):
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
                                "enforce_perf": False,
                                "perf": {"decode_t/s/u": 100.0},
                                "accuracy": {"top1": 90.0},
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 3}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "top1" in result.stdout
    assert "::error::" in result.stdout


def test_schema_rejects_non_bool_enforce_perf(tmp_path):
    validator = _load_validator_module()
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {
                                "status": "active",
                                "enforce_perf": "no",
                                "perf": {},
                                "accuracy": {},
                            }
                        ]
                    }
                }
            }
        },
    }
    errors = validator._validate_targets_schema(targets)
    assert any("enforce_perf" in error for error in errors)


def _write_rtf_run(path: Path, rtf: float) -> None:
    """A speech run that reports the real-time factor from a plain 'inference' step."""
    path.write_text(
        json.dumps(
            {
                "ml_model_name": "demo-model",
                "batch_size": 1,
                "input_sequence_length": 128,
                "measurements": [{"step_name": "inference", "name": "rtf", "value": rtf}],
            }
        ),
        encoding="utf-8",
    )


def test_validate_perf_targets_rtf_within_tolerance_passes(tmp_path):
    """rtf is a perf metric; a value inside the +/- tolerance band (including below target) passes."""
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    # Target rtf=0.5, tolerance 0.1 -> band [0.45, 0.55]. Measured 0.47 is below target (better)
    # yet still inside the band, so it must pass: lower-is-better values are accepted.
    _write_rtf_run(tmp_path / "generated/benchmark_data/complete_run_1.json", rtf=0.47)
    _write_single_target(tmp_path, perf={"rtf": 0.5, "rtf_tolerance": 0.1})

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rtf" in result.stdout
    assert "pass" in result.stdout


def test_validate_perf_targets_rtf_regression_above_target_fails(tmp_path):
    """Exceeding the rtf target (slower than real time) is a regression for a lower-is-better metric."""
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    # Measured 0.9 > upper_bound 0.55 -> fails, and the note must read "regression" (not the
    # "update target" wording a higher-is-better metric would use when it overshoots).
    _write_rtf_run(tmp_path / "generated/benchmark_data/complete_run_1.json", rtf=0.9)
    _write_single_target(tmp_path, perf={"rtf": 0.5, "rtf_tolerance": 0.1})

    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "rtf" in result.stdout
    assert "regression" in result.stdout


def test_validate_perf_targets_rtf_far_below_target_flags_stale_target(tmp_path):
    """A value far under the band is still out-of-band, but for rtf it means the target is stale."""
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    # Measured 0.1 < lower_bound 0.45 -> out of band; for a lower-is-better metric this is
    # "much better than expected, update target", confirming the direction classification.
    _write_rtf_run(tmp_path / "generated/benchmark_data/complete_run_1.json", rtf=0.1)
    _write_single_target(tmp_path, perf={"rtf": 0.5, "rtf_tolerance": 0.1})

    result = _run_validator(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "much better than expected, update target" in result.stdout


def test_validate_perf_targets_rtf_report_only_does_not_fail(tmp_path):
    """enforce_perf: false makes an rtf regression report-only, matching other perf metrics."""
    (tmp_path / "generated/benchmark_data").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "tests/pipeline_reorg").mkdir(parents=True)

    _write_rtf_run(tmp_path / "generated/benchmark_data/complete_run_1.json", rtf=0.9)
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
                                "enforce_perf": False,
                                "perf": {"rtf": 0.5, "rtf_tolerance": 0.1},
                                "accuracy": {},
                            }
                        ]
                    }
                },
            }
        },
    }
    (tmp_path / "models/model_targets.yaml").write_text(yaml.safe_dump(targets), encoding="utf-8")
    tests_yaml = [{"model": "demo-model", "skus": {"wh_n150": {"tier": 3}}, "team": "models"}]
    (tmp_path / "tests/pipeline_reorg/models_e2e_tests.yaml").write_text(yaml.safe_dump(tests_yaml), encoding="utf-8")

    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "report-only" in result.stdout
    assert "::error::" not in result.stdout


def test_report_generation_rate_uses_explicit_batch_size(monkeypatch):
    """When batch_size is passed it is recorded verbatim; count only drives the rate numerator."""
    module = _load_common_demo_utils_module()
    calls = []
    monkeypatch.setattr(module, "report_vision_fps", lambda *args: calls.append(args))

    # A video model: 48 frames in 2 s -> 24 fps, but the workload batch_size is 1, not 48.
    module.report_generation_rate("video-model", 2.0, count=48, batch_size=1)
    assert calls == [("video-model", 24.0, 1)]


def test_report_generation_rate_defaults_batch_size_to_count(monkeypatch):
    """Backwards compatibility: without an explicit batch_size, count is reused as before."""
    module = _load_common_demo_utils_module()
    calls = []
    monkeypatch.setattr(module, "report_vision_fps", lambda *args: calls.append(args))

    module.report_generation_rate("classifier", 4.0, count=16)
    assert calls == [("classifier", 4.0, 16)]


def test_report_generation_rate_preserves_fractional_count(monkeypatch):
    """A fractional count is used exactly for the rate; the explicit batch_size is not truncated."""
    module = _load_common_demo_utils_module()
    calls = []
    monkeypatch.setattr(module, "report_vision_fps", lambda *args: calls.append(args))

    module.report_generation_rate("speech", 2.0, count=3.0, batch_size=2)
    assert calls == [("speech", 1.5, 2)]


def test_report_generation_rate_noop_on_nonpositive_inputs(monkeypatch):
    """Guard clause: non-positive elapsed time or count emits nothing."""
    module = _load_common_demo_utils_module()
    calls = []
    monkeypatch.setattr(module, "report_vision_fps", lambda *args: calls.append(args))

    module.report_generation_rate("m", 0.0, count=10)
    module.report_generation_rate("m", -1.0, count=10, batch_size=4)
    module.report_generation_rate("m", 2.0, count=0)
    assert calls == []


def test_report_e2e_metrics_noop_outside_ci(monkeypatch):
    """Outside CI the helper is a no-op and writes no partial run."""
    module = _load_common_demo_utils_module()
    saved: list = []
    _install_benchmark_stub(monkeypatch, is_ci=False, saved=saved)

    module.report_e2e_metrics("demo-model", {"decode_t/s": 100.0}, batch_size=1)
    assert saved == []


def test_report_e2e_metrics_builds_known_measurements_in_ci(monkeypatch):
    """In CI the helper maps each known metric to its (step, measurement) pair and run metadata."""
    module = _load_common_demo_utils_module()
    saved: list = []
    _install_benchmark_stub(monkeypatch, is_ci=True, saved=saved)

    module.report_e2e_metrics(
        "mistral-small-3.1-24b",
        {"decode_t/s": 50.0, "decode_t/s/u": 50.0, "prefill_t/s": 200.0},
        batch_size=1,
        seq_len=128,
        model_type="vlm",
    )

    assert len(saved) == 1
    record = saved[0]
    assert record["ml_model_name"] == "mistral-small-3.1-24b"
    assert record["ml_model_type"] == "vlm"
    assert record["batch_size"] == 1
    assert record["input_sequence_length"] == 128
    measured = {(step, name): value for step, name, value in record["measurements"]}
    assert measured[("inference_decode", "tokens/s")] == 50.0
    assert measured[("inference_decode", "tokens/s/user")] == 50.0
    assert measured[("inference_prefill", "tokens/s")] == 200.0


def test_report_e2e_metrics_rejects_unknown_metric(monkeypatch):
    """An unknown metric name is a programming error and must raise rather than silently drop."""
    module = _load_common_demo_utils_module()
    saved: list = []
    _install_benchmark_stub(monkeypatch, is_ci=True, saved=saved)

    try:
        module.report_e2e_metrics("demo-model", {"not_a_real_metric": 1.0}, batch_size=1)
        assert False, "Expected ValueError for unknown e2e metric"
    except ValueError as exc:
        assert "not_a_real_metric" in str(exc)


def test_report_e2e_metrics_noop_on_empty_metrics(monkeypatch):
    """No metrics means nothing to report, even in CI."""
    module = _load_common_demo_utils_module()
    saved: list = []
    _install_benchmark_stub(monkeypatch, is_ci=True, saved=saved)

    module.report_e2e_metrics("demo-model", {}, batch_size=1)
    assert saved == []
