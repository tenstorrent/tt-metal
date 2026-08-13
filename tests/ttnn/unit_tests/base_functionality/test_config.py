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
    original = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("_") and key != "report_path" and not callable(getattr(config, key))
    }
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
    with expect_error(RuntimeError, "Unknown configuration key"):
        ttnn.CONFIG.apply_json_overrides('{"no_such_key": 1}')
    ttnn.CONFIG.apply_json_overrides('{"no_such_key": 1}', strict=False)


def test_apply_json_overrides_rejects_non_object(expect_error):
    with expect_error(RuntimeError, "must be a JSON object"):
        ttnn.CONFIG.apply_json_overrides('["validate_program_args"]')


def test_load_config_from_dictionary(expect_error):
    ttnn.load_config_from_dictionary({"enable_model_cache": True})
    assert ttnn.CONFIG.enable_model_cache is True
    with expect_error(RuntimeError, "Unknown configuration key"):
        ttnn.load_config_from_dictionary({"no_such_key": 1})
    ttnn.load_config_from_dictionary({"no_such_key": 1}, from_file=True)


def test_config_file_round_trip(tmp_path):
    path = tmp_path / "config.json"
    ttnn.save_config_to_json_file(path)
    saved_report_name = ttnn.CONFIG.report_name
    ttnn.CONFIG.enable_model_cache = True
    ttnn.load_config_from_json_file(path)
    assert ttnn.CONFIG.enable_model_cache is False
    assert ttnn.CONFIG.report_name == saved_report_name


def load_raw_ttnn_module(extra_env):
    """Load _ttnn directly, bypassing ttnn/__init__.py, as a pure C++ consumer would."""
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_ttnn', {ttnn._ttnn.__file__!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(m.CONFIG.validate_program_args)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, **extra_env}, capture_output=True, text=True, timeout=120
    )


def test_env_overrides_apply_without_python_init():
    result = load_raw_ttnn_module({"TTNN_CONFIG_OVERRIDES": '{"validate_program_args": true}'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_env_overrides_unknown_key_fails_at_load():
    result = load_raw_ttnn_module({"TTNN_CONFIG_OVERRIDES": '{"no_such_key": true}'})
    assert result.returncode != 0
