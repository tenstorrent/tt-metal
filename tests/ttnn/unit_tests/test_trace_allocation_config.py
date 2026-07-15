# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

CONFIG_PATH = Path(__file__).parents[3] / "ttnn" / "ttnn" / "trace_allocation_config.py"
ENV_NAMES = (
    "TT_METAL_TRACE_ALLOC_TRACKING",
    "TT_METAL_TRACE_ALLOC_TRACEBACKS",
    "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH",
)


def read_config(env_overrides):
    env = os.environ.copy()
    for name in ENV_NAMES:
        env.pop(name, None)
    env.update(env_overrides)
    script = f"""
import json
import runpy
config = runpy.run_path({str(CONFIG_PATH)!r})
print(json.dumps({{
    'tracking': config['TRACE_ALLOC_TRACKING'],
    'diagnostics': config['TRACE_ALLOC_DIAGNOSTICS'],
    'depth': config['TRACE_ALLOC_REFERRER_DEPTH'],
}}))
"""
    result = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, {"tracking": False, "diagnostics": False, "depth": 10}),
        (
            {"TT_METAL_TRACE_ALLOC_TRACEBACKS": "1"},
            {"tracking": False, "diagnostics": False, "depth": 10},
        ),
        (
            {"TT_METAL_TRACE_ALLOC_TRACKING": "1"},
            {"tracking": True, "diagnostics": False, "depth": 10},
        ),
        (
            {"TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_TRACE_ALLOC_TRACEBACKS": "1"},
            {"tracking": True, "diagnostics": True, "depth": 10},
        ),
        (
            {
                "TT_METAL_TRACE_ALLOC_TRACKING": "1",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS": "1",
                "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH": "4",
            },
            {"tracking": True, "diagnostics": True, "depth": 4},
        ),
    ],
)
def test_trace_allocation_config_is_captured_at_startup(env, expected):
    assert read_config(env) == expected


def test_invalid_referrer_depth_uses_default():
    assert (
        read_config(
            {
                "TT_METAL_TRACE_ALLOC_TRACKING": "1",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS": "1",
                "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH": "invalid",
            }
        )["depth"]
        == 10
    )


def test_disabled_tracking_uses_direct_execute_trace_binding():
    env = os.environ.copy()
    for name in ENV_NAMES:
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", "import ttnn; print(ttnn.execute_trace is ttnn._ttnn_execute_trace)"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip().endswith("True")
