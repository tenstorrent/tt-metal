# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device perf: Tracy profile of TTNN ``demo.run_inference`` (nested pytest on runner module).

The forward run lives in ``test_perf_ttnn_lingbot_va_runner.py``.
"""

from __future__ import annotations

import warnings

# Before any import that may pull ttnn/SWIG (so Python does not emit these during import).
warnings.filterwarnings("ignore", message=r".*SwigPy(Packed|Object).*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)


import pytest
from tracy.process_model_log import run_device_profiler

import models
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(SwigPy|swigvarlink).*:DeprecationWarning",
)


def _run_device_profiler_op_support_count(*args, **kwargs):
    # Default cap is low; Lingbot-VA graphs exceed it without this override.
    kwargs.setdefault("op_support_count", 7500)
    return run_device_profiler(*args, **kwargs)


models.perf.device_perf_utils.run_device_profiler = _run_device_profiler_op_support_count


@pytest.mark.parametrize(
    "batch_size, model_name",
    [
        (1, "ttnn_lingbot_va"),
    ],
)
@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_lingbot_va(batch_size, model_name):
    subdir = model_name
    num_iterations = 1
    margin = 0.16

    command = "pytest models/experimental/lingbot_va/tests/perf/test_perf_ttnn_lingbot_va_runner.py::test_lingbot_va_ttnn_forward_run"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: 0.49}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
