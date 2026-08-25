# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

import pytest

ENV_NAMES = (
    "TT_METAL_TRACE_ALLOC_TRACKING",
    "TT_METAL_TRACE_ALLOC_TRACEBACKS",
    "TT_METAL_TRACE_ALLOC_REFERRER_DEPTH",
    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
)

READ_CONFIG_SCRIPT = """
import json
from ttnn.tools.trace_allocation_tracker import (
    TRACE_ALLOC_DIAGNOSTICS,
    TRACE_ALLOC_REFERRER_DEPTH,
    TRACE_ALLOC_TRACKING,
)
print(json.dumps({
    'tracking': TRACE_ALLOC_TRACKING,
    'diagnostics': TRACE_ALLOC_DIAGNOSTICS,
    'depth': TRACE_ALLOC_REFERRER_DEPTH,
}))
"""


def read_config(env_overrides):
    env = os.environ.copy()
    for name in ENV_NAMES:
        env.pop(name, None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", READ_CONFIG_SCRIPT], env=env, text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout.splitlines()[-1])


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
def test_trace_allocation_tracker_config_is_captured_at_startup(env, expected):
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
