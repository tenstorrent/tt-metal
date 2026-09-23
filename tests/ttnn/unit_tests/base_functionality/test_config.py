# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import subprocess
import sys

import pytest

import ttnn


@pytest.fixture(autouse=True)
def restore_config():
    config = ttnn._ttnn.CONFIG
    original = {key: getattr(config, key) for key in ttnn.Config.keys()}
    yield
    for key, value in original.items():
        setattr(config, key, value)


def test_apply_json_overrides():
    ttnn.CONFIG.apply_json_overrides(
        json.dumps(
            {
                "validate_program_args": True,
                "comparison_mode_pcc": 0.5,
                "tmp_dir": "/tmp/ttnn-test",
                "report_name": "config test",
            }
        )
    )
    assert ttnn.CONFIG.validate_program_args is True
    assert ttnn.CONFIG.comparison_mode_pcc == pytest.approx(0.5)
    assert ttnn.CONFIG.tmp_dir == pathlib.Path("/tmp/ttnn-test")
    assert ttnn.CONFIG.report_name == pathlib.Path("config test")


def test_apply_json_overrides_null_clears_optional():
    ttnn.CONFIG.apply_json_overrides('{"report_name": "something"}')
    ttnn.CONFIG.apply_json_overrides('{"report_name": null}')
    assert ttnn.CONFIG.report_name is None


def test_apply_json_overrides_unknown_key(expect_error):
    with expect_error(RuntimeError, "Unknown configuration key: no_such_key"):
        ttnn.CONFIG.apply_json_overrides('{"no_such_key": 1}')
    ttnn.CONFIG.apply_json_overrides('{"no_such_key": 1}', strict=False)


def test_apply_json_overrides_names_the_source(expect_error):
    with expect_error(RuntimeError, "from /some/config.json"):
        ttnn.CONFIG.apply_json_overrides('{"no_such_key": 1}', source="/some/config.json")


def test_apply_json_overrides_rejects_non_object(expect_error):
    with expect_error(RuntimeError, "must be a JSON object"):
        ttnn.CONFIG.apply_json_overrides('["validate_program_args"]')


def test_apply_json_overrides_rejects_malformed_json(expect_error):
    with expect_error(RuntimeError, "not valid JSON"):
        ttnn.CONFIG.apply_json_overrides('{"validate_program_args": true,}')


@pytest.mark.parametrize(
    "bad",
    [
        '{"enable_model_cache": true, "no_such_key": 1}',
        '{"enable_model_cache": true, "comparison_mode_pcc": "0.5"}',
    ],
)
def test_strict_failure_applies_nothing(bad, expect_error):
    ttnn.CONFIG.enable_model_cache = False
    with expect_error(RuntimeError, "configuration key"):
        ttnn.CONFIG.apply_json_overrides(bad)
    assert ttnn.CONFIG.enable_model_cache is False


def test_non_strict_keeps_the_keys_that_parsed():
    ttnn.CONFIG.enable_model_cache = False
    pcc = ttnn.CONFIG.comparison_mode_pcc
    ttnn.CONFIG.apply_json_overrides(
        '{"enable_model_cache": true, "comparison_mode_pcc": "0.5", "no_such_key": 1}', strict=False
    )
    assert ttnn.CONFIG.enable_model_cache is True
    assert ttnn.CONFIG.comparison_mode_pcc == pytest.approx(pcc)


def test_load_config_from_dictionary(expect_error):
    ttnn.load_config_from_dictionary({"enable_model_cache": True})
    assert ttnn.CONFIG.enable_model_cache is True
    with expect_error(ValueError, "Unknown configuration key"):
        ttnn.load_config_from_dictionary({"no_such_key": 1})
    ttnn.load_config_from_dictionary({"no_such_key": 1}, from_file=True)


def test_saved_config_holds_exactly_the_overridable_keys(tmp_path):
    path = tmp_path / "config.json"
    ttnn.save_config_to_json_file(path)
    assert set(json.loads(path.read_text())) == set(ttnn.Config.keys())


def test_config_file_round_trip(tmp_path):
    path = tmp_path / "config.json"
    ttnn.save_config_to_json_file(path)
    saved_report_name = ttnn.CONFIG.report_name
    ttnn.CONFIG.enable_model_cache = True
    ttnn.load_config_from_json_file(path)
    assert ttnn.CONFIG.enable_model_cache is False
    assert ttnn.CONFIG.report_name == saved_report_name


def run_python(script, extra_env):
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **extra_env},
        capture_output=True,
        text=True,
        timeout=600,
    )


def load_raw_ttnn_module(extra_env):
    """Load _ttnn directly, bypassing ttnn/__init__.py, as a pure C++ consumer would."""
    return run_python(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_ttnn', {ttnn._ttnn.__file__!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(m.CONFIG.validate_program_args)\n",
        extra_env,
    )


def test_env_overrides_apply_without_python_init():
    result = load_raw_ttnn_module({"TTNN_CONFIG_OVERRIDES": '{"validate_program_args": true}'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ('{"no_such_key": true}', "Unknown configuration key: no_such_key"),
        ('{"validate_program_args": true,}', "not valid JSON"),
        ('{"validate_program_args": 1}', "Bad value for configuration key validate_program_args"),
    ],
)
def test_bad_env_overrides_exit_cleanly_at_load(overrides, message):
    result = load_raw_ttnn_module({"TTNN_CONFIG_OVERRIDES": overrides})
    # Exit code, not a signal: the load-time throw must not reach std::terminate/SIGABRT.
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    assert "TTNN_CONFIG_OVERRIDES" in result.stdout


def test_config_file_applies_without_python_init(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"validate_program_args": True}))
    result = load_raw_ttnn_module({"TTNN_CONFIG_PATH": str(path)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_missing_config_file_is_seeded_with_defaults_without_python_init(tmp_path):
    path = tmp_path / "nested" / "config.json"
    result = load_raw_ttnn_module(
        {"TTNN_CONFIG_PATH": str(path), "TTNN_CONFIG_OVERRIDES": '{"validate_program_args": true}'}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads(path.read_text())
    assert set(saved) == set(ttnn.Config.keys())
    # The env override applies to the process but must not be baked into the file as a default.
    assert saved["validate_program_args"] is False
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_stale_config_file_key_warns_and_keeps_the_rest(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"report_path": "generated", "validate_program_args": True}))
    result = load_raw_ttnn_module({"TTNN_CONFIG_PATH": str(path)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unknown configuration key: report_path" in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_corrupt_config_file_does_not_fail_the_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    result = load_raw_ttnn_module({"TTNN_CONFIG_PATH": str(path)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Failed to load ttnn configuration" in result.stdout


def test_env_overrides_take_precedence_over_the_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enable_model_cache": True, "validate_program_args": True}))
    result = run_python(
        "import ttnn; print(ttnn.CONFIG.enable_model_cache, ttnn.CONFIG.validate_program_args)",
        {"TTNN_CONFIG_PATH": str(path), "TTNN_CONFIG_OVERRIDES": '{"enable_model_cache": false}'},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False True"


def test_env_overrides_are_validated_once_per_process():
    result = run_python("import ttnn", {"TTNN_CONFIG_OVERRIDES": '{"enable_logging": true}'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Logging cannot be enabled in fast runtime mode") == 1
