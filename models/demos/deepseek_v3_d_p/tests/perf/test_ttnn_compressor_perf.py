# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Device-perf test for the DeepSeek-V4 HCA compressor (prefill), single WH device.

Runs the PCC test under the device profiler and measures the device kernel time of the
compressor forward — the ops between the HCA_START / HCA_END signposts, excluding the
one-time weight/input tilize dispatched at construction before HCA_START. Only the two
projections (kv_proj / gate_proj) are on device today; the baseline grows as more stages
migrate host->device.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_compressor.py::test_hca_compressor"
_CMD = f"pytest {_TEST_PATH} -k 'b1-seq512'"


@pytest.mark.timeout(0)
def test_hca_compressor_perf():
    run_model_device_perf_test_with_merge(
        command=_CMD,
        expected_device_perf_ns_per_iteration=255_569,
        subdir="deepseek_v4_hca_compressor",
        model_name="deepseek_v4_hca_compressor",
        num_iterations=1,
        batch_size=1,
        margin=0.03,
        between_signposts=("HCA_START", "HCA_END"),
        comments="seq512_hca_compressor_prefill",
    )
