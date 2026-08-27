# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf wrappers for local Fabric2D, 4x4 subtorus, and production TorusXY workloads."""

import os
import re

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import _is_galaxy_env, run_model_device_perf_test_with_merge
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"

_IGNORE_POWER = os.environ.get("DS_PERF_IGNORE_POWER") == "1"

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
        pytest.param(
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4-2link and layer3 and gate_device and no_ref and isl_1280' --wrapper-invocation",
            10_963_542,  # Measured 2026-08-27 at 640 tokens/chip on the CI LoudBox (bh_loudbox, 8xP150),
            # run 33082257984 job bh_lb_DeepSeek_PREFILL_PERF. Single observation. Supersedes
            # 14_179_641, a 14-run mean on bh-lb-15 (8xp150b) -- the two boxes disagree by 23%, so
            # this gate has to be cut on the CI runner, not a dev LoudBox.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer0 and gate_device and no_ref and isl_5120' --wrapper-invocation",
            5_435_504,  # Measured 2026-08-22 on the 14kW BH galaxy bh-glx-110-c04u02, 8x4 TorusXY certified
            # (DDR 16000 nominal, high power).
            # Two runs 5.421 / 5.450 ms, spread 0.54% -- well inside the 3% band.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_torus_xy",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-xy-8x4 and layer3 and gate_device and no_ref and isl_5120' --wrapper-invocation",
            13_674_937,  # Measured 2026-08-22 on the 14kW BH galaxy bh-glx-110-c04u02, 8x4 TorusXY certified
            # (DDR 16000 nominal, high power).
            # Two runs 13.558 / 13.792 ms, spread 1.71% -- inside the 3% band, the widest of the five.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_torus_xy",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer0 and gate_device and no_ref and isl_2560' --wrapper-invocation",
            4_913_214,  # Re-measured 2026-08-27 at 640 tokens/chip on the 14kW galaxy bh-glx-120-d08u02
            # (is_high_power). Mean of 2 runs, 4.9120-4.9144 ms, 0.05% peak to peak.
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer0_dense_torus_y",
            1,
            1,
            0.03,
            "subtorus_4x4_layer0_dense_real_weights_torus_y",
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-y-4x4 and layer3 and gate_device and no_ref and isl_2560' --wrapper-invocation",
            15_570_232,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_y",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-x-4x4 and layer3 and gate_device and no_ref and isl_2560' --wrapper-invocation",
            54_804_819,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_x",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_torus_x",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
        pytest.param(
            f"pytest {_TEST_PATH} -k 'torus-xy-4x4 and layer3 and gate_device and no_ref and isl_2560' --wrapper-invocation",
            52_978_544,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_4x4_layer3_moe_torus_xy",
            1,
            1,
            0.03,
            "subtorus_4x4_layer3_moe_128experts_8perchip_hostgate_torus_xy",
            marks=_SUBTORUS_4X4_HOSTGATE_SKIP,
        ),
    ],
    ids=[
        "block_2x4_layer3_moe_fabric2d",
        "block_8x4_layer0_dense_torus_xy",
        "block_8x4_layer3_moe_torus_xy",
        "block_4x4_layer0_dense_torus_y",
        "block_4x4_layer3_moe_torus_y",
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
    if ("_8x4_" in model_name or "_4x4_" in model_name) and not (is_high_power() or _IGNORE_POWER):
        pytest.skip(
            "galaxy perf baselines are cut on a >=130W TDP host; an 8kW galaxy measures differently. "
            "DS_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only"
        )

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
