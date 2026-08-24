# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf wrappers for local Fabric2D, 4x4 subtorus, and production TorusXY workloads."""

import os
import re

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import _is_galaxy_env, run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
_SUBTORUS_Y4_ENV = {
    "TT_VISIBLE_DEVICES": "2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31",
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/"
        "single_bh_galaxy_subtorus_y4_graph_descriptor.textproto",
    ),
}
_SUBTORUS_X4_ENV = {
    "TT_VISIBLE_DEVICES": _SUBTORUS_Y4_ENV["TT_VISIBLE_DEVICES"],
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/"
        "single_bh_galaxy_subtorus_x4_graph_descriptor.textproto",
    ),
}
_SUBTORUS_XY4_ENV = {
    "TT_VISIBLE_DEVICES": _SUBTORUS_Y4_ENV["TT_VISIBLE_DEVICES"],
    "TT_MESH_GRAPH_DESC_PATH": os.path.join(
        _REPO_ROOT,
        "models/demos/deepseek_v3_d_p/experimental_descriptors/"
        "single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto",
    ),
}

_SUBTORUS_4X4_HOSTGATE_SKIP = pytest.mark.skip(
    reason="4x4 subtorus MoE uses the 128-expert host gate; its device-perf is host-stall-dominated "
    "and needs a new baseline before the gate can be re-enabled"
)


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4'",
            30_330_761,  # Recalibrated 2026-08-21 on this LoudBox with the routed experts folded
            # into one program. Single run, where the previous value was a mean of three.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d",
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
        (
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer0 and gate_device and no_ref and isl_12k8'",
            17_978_418,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer0_dense_torus_y",
            1,
            1,
            0.03,
            "subtorus_4x4_layer0_dense_real_weights_torus_y_isl12k8",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            56_528_886,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer3 and gate_device and no_ref and isl_2k56'",
            15_570_232,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y_isl2k56",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl2k56",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-x-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            54_804_819,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_x",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8_torus_x",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-xy-4x4 and layer3 and gate_device and no_ref and isl_12k8'",
            52_978_544,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_isl12k8_torus_xy",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
    ],
    ids=[
        "block_2x4_layer3_moe_fabric2d",
        "block_8x4_layer0_dense_torus_xy",
        "block_8x4_layer3_moe_torus_xy",
        "block_4x4_layer0_dense_torus_y",
        "block_4x4_layer3_moe_torus_y",
        "block_4x4_layer3_moe_torus_y_isl2k56",
        "block_4x4_layer3_moe_torus_x",
        "block_4x4_layer3_moe_torus_xy",
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
    if "_8x4_" in model_name and "torus_xy" in model_name:
        if not _is_galaxy_env():
            pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
        if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
            pytest.fail("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")

    extra_env = None
    if "_4x4_" in model_name:
        carve_env_by_axis = {"x": _SUBTORUS_X4_ENV, "xy": _SUBTORUS_XY4_ENV, "y": _SUBTORUS_Y4_ENV}
        match = re.search(r"_torus_(xy|x|y)(?:_|$)", model_name)
        assert match is not None, f"4x4 perf entry {model_name!r} has no _torus_<axis> token"
        extra_env = dict(carve_env_by_axis[match.group(1)])
        extra_env["DS_4X4_FULL_EXPERTS"] = ""

    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
        extra_env=extra_env,
    )
