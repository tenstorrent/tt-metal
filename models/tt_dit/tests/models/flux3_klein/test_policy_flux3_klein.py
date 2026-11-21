# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

from ....pipelines.flux3_klein import pipeline_flux3_klein, policy, scheduler


def test_pipeline_takes_the_policy_values():
    assert tuple((size, size) for size in pipeline_flux3_klein.SIZES) == policy.FLUX3_KLEIN_SERVING_SIZES
    assert (pipeline_flux3_klein.DEFAULT_SIZE,) * 2 == policy.FLUX3_KLEIN_DEFAULT_SIZE
    assert pipeline_flux3_klein.MAX_REFERENCES == policy.FLUX3_KLEIN_MAX_REFERENCES
    assert scheduler.STEPS["distilled"] == policy.FLUX3_KLEIN_NUM_INFERENCE_STEPS


@pytest.mark.parametrize(
    "upsampler_resident, expected",
    [
        (False, ((1024, 1024), (2048, 2048), (4096, 4096))),
        (True, ((1024, 1024), (2048, 2048))),
    ],
)
def test_served_resolutions(upsampler_resident, expected):
    assert policy.served_resolutions(upsampler_resident=upsampler_resident) == expected


def test_validate_size_accepts_a_served_size():
    policy.validate_size(width=4096, height=4096, upsampler_resident=False)
    policy.validate_size(width=2048, height=2048, upsampler_resident=True)


@pytest.mark.parametrize(
    "width, height, upsampler_resident, message",
    [
        (2048, 1024, False, "unsupported size 2048x1024; this deployment serves"),
        (4096, 4096, True, "unsupported size 4096x4096 with the prompt upsampler resident; this deployment serves"),
    ],
)
def test_validate_size_rejects_an_unserved_size(width, height, upsampler_resident, message, expect_error):
    with expect_error(ValueError, message):
        policy.validate_size(width=width, height=height, upsampler_resident=upsampler_resident)


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
