# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json

import pytest

import ttnn
from ttnn.decorators import FastOperation


@pytest.mark.parametrize("fast,comparison,logging", itertools.product((False, True), repeat=3))
def test_requires_slow_runtime_uses_current_config(fast, comparison, logging):
    config = ttnn.Config()
    config.apply_json_overrides(
        json.dumps(
            {
                "enable_fast_runtime_mode": fast,
                "enable_comparison_mode": comparison,
                "enable_logging": logging,
            }
        )
    )
    assert config.requires_slow_runtime is (not fast or comparison or logging)


def test_existing_fast_operation_observes_config_changes(monkeypatch):
    config = ttnn.Config()
    monkeypatch.setattr(ttnn, "CONFIG", config)
    monkeypatch.setattr(ttnn.graph, "is_python_io_recording_enabled", lambda: False)
    calls = []

    def fast_function(value):
        calls.append("fast")
        return value + 1

    def slow_function(value):
        calls.append("slow")
        return value + 2

    operation = FastOperation(
        python_fully_qualified_name="ttnn.test_live_config",
        function=fast_function,
        preprocess_golden_function_inputs=None,
        golden_function=None,
        postprocess_golden_function_outputs=None,
        is_cpp_operation=False,
        is_experimental=False,
    )
    monkeypatch.setattr(operation, "_slow_operation_instance", lambda: slow_function)

    assert operation(10) == 11
    for flag, enabled, restored in (
        ("enable_fast_runtime_mode", False, True),
        ("enable_comparison_mode", True, False),
        ("enable_logging", True, False),
    ):
        setattr(config, flag, enabled)
        assert operation(10) == 12
        setattr(config, flag, restored)
        assert operation(10) == 11

    config.apply_json_overrides('{"enable_comparison_mode": true, "enable_logging": true}')
    assert operation(10) == 12
    config.apply_json_overrides('{"enable_comparison_mode": false}')
    assert operation(10) == 12  # Logging still independently requires the slow path.
    config.apply_json_overrides('{"enable_logging": false}')
    assert operation(10) == 11
    assert calls == ["fast", "slow", "fast", "slow", "fast", "slow", "fast", "slow", "slow", "fast"]


def test_requires_slow_runtime_is_derived_and_read_only(expect_error):
    config = ttnn.Config()
    assert "requires_slow_runtime" not in ttnn.Config.keys()
    with expect_error(AttributeError, "can.t set attribute"):
        config.requires_slow_runtime = True
    config.enable_logging = True
    clone = ttnn.Config(config)
    assert clone.requires_slow_runtime is True
    clone.enable_logging = False
    assert clone.requires_slow_runtime is False
    assert config.requires_slow_runtime is True
