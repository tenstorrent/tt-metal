# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

from ....pipelines.ideogram4 import pipeline_ideogram4, policy


def test_config_default_takes_the_served_resolutions():
    for mesh_shape in pipeline_ideogram4._PRESETS:
        config = pipeline_ideogram4.Ideogram4PipelineConfig.default(mesh_shape=mesh_shape)
        assert config.resolutions == policy.served_resolutions(mesh_shape), mesh_shape


@pytest.mark.parametrize(
    "mesh_shape, expected",
    [
        ((2, 4), policy.IDEOGRAM4_SERVING_SIZES),
        ((4, 8), policy.IDEOGRAM4_SERVING_SIZES),
        ((1, 4), ()),
    ],
)
def test_served_resolutions(mesh_shape, expected):
    assert policy.served_resolutions(mesh_shape) == expected


def _counter(tokens):
    return lambda prompt: tokens


def test_validate_request_accepts_a_served_size_within_the_limit():
    policy.validate_request(mesh_shape=(4, 8), width=2048, height=1152, prompt="p", count_text_tokens=_counter(1))
    policy.validate_request(mesh_shape=(2, 4), width=1360, height=2048, prompt="p", count_text_tokens=_counter(1))


@pytest.mark.parametrize("mesh_shape", [(4, 8), (1, 4)])
def test_validate_request_rejects_an_unserved_size(mesh_shape, expect_error):
    with expect_error(ValueError, "unsupported size 1024x1024; this deployment serves"):
        policy.validate_request(
            mesh_shape=mesh_shape, width=1024, height=1024, prompt="p", count_text_tokens=_counter(1)
        )


def test_validate_prompt_accepts_the_limit():
    policy.validate_prompt("p", _counter(policy.IDEOGRAM4_MAX_TEXT_TOKENS))


def test_validate_prompt_rejects_an_over_long_prompt(expect_error):
    with expect_error(ValueError, "prompt is 2049 tokens, over the 2048-token limit"):
        policy.validate_prompt("p", _counter(policy.IDEOGRAM4_MAX_TEXT_TOKENS + 1))


def test_policy_imports_no_ttnn():
    # Loaded by path: importing it as a package module runs models/tt_dit/pipelines/__init__.py,
    # which imports ttnn.
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('policy', {policy.__file__!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print('ttnn' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
