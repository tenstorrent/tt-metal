# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf wrappers for local Fabric2D and production TorusXY block workloads.

Legacy Fabric1d, TorusX, 4x4 sub-torus, and unwrapped 8x4 rows are redundant after migration.
TorusY is retained only as an unscheduled axis diagnostic; TorusXY owns production performance.
"""

import os

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4'",
            None,  # Record-only until calibrated on Fabric2D; the old 35,655,993 ns value was Fabric1D.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d_2link",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d_2link_record_only_pending_calibration",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4 and not 2link and layer3 and gate_device and no_ref and isl_6k4'",
            48_977_160,  # Calibrated 2026-07-29 from three BH LoudBox Fabric2D runs.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-y-diagnostic-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            25_236_993,  # Calibrated 2026-06-26 on BH Galaxy 110-c78, TorusY, real weights.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_y_diagnostic",
            1,
            1,
            0.03,
            "unscheduled_glx_8x4_layer0_dense_torus_y_diagnostic",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-y-diagnostic-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            67_193_413,  # Calibrated 2026-06-26 on BH Galaxy 110-c78, TorusY, real weights.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_y_diagnostic",
            1,
            1,
            0.03,
            "unscheduled_glx_8x4_layer3_moe_torus_y_diagnostic",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            18_157_603,  # Calibrated 2026-07-01 on BH Galaxy 110-c910, TorusXY, real weights.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_torus_xy",
        ),
        (
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            60_634_662,  # Calibrated 2026-07-01 on BH Galaxy 110-c910, TorusXY, real weights.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_torus_xy",
        ),
    ],
    ids=[
        "block_2x4_layer3_moe_fabric2d_2link",
        "block_2x4_layer3_moe_fabric2d",
        "diagnostic_block_8x4_layer0_dense_torus_y",
        "diagnostic_block_8x4_layer3_moe_torus_y",
        "block_8x4_layer0_dense_torus_xy",
        "block_8x4_layer3_moe_torus_xy",
    ],
)
@pytest.mark.timeout(0)
def test_deepseek_v3_prefill_block_perf(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
):
    if "torus" in model_name and (os.getenv("CI") == "true" or "TT_GH_CI_INFRA" in os.environ):
        certified_torus_xy = "torus_xy" in model_name and os.getenv("PREFILL_TORUS_XY_CERTIFIED") == "1"
        if not certified_torus_xy:
            pytest.skip("Torus perf requires the production TorusXY case on a cabling-certified Galaxy")

    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
