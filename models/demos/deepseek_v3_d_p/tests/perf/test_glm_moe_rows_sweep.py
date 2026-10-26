# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.3 MoE cost vs tokens per call: the measurement behind packed (batched) prefill forwards.

A packed forward of F chunks runs ONE MoE over F x 640 rows per chip instead of F MoEs over 640. The
routed experts' weights are read once per call, so if the MoE is bound by that read its time stays
nearly flat as rows grow, and packing pays; if it scales with rows, packing saves only fixed overhead.

Each row times one traced replay of the MoE forward (the test_kimi_moe_perf harness: untimed compile
pass, capture, then ``GLM_MOE_SWEEP_REPLAYS`` timed replays, median reported). Record only, no gate.
Select with ``-k loudbox`` / ``-k galaxy`` and ``-k x1`` / ``x2`` / ``x4``.
"""

import os
import statistics
import time

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import routed_expert_combine_overlap_arg
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

_DISPATCH_BUFFER_CAPACITY_FACTOR = 8  # as test_glm_moe
_TRACE_REGION_SIZE = 32 * 1024 * 1024
_REPLAYS = int(os.environ.get("GLM_MOE_SWEEP_REPLAYS", "5"))

_MESHES = [
    ("galaxy-torus-xy-8x4", (8, 4), torus_xy_device_params, "mesh-8x4"),
    ("loudbox-fabric2d-2x4", (2, 4), fabric2d_device_params, "mesh-2x4"),
]


def _replay_ns(mesh_device, copy_tid, controller):
    """Wall-clock of one segmented replay; the input copy runs before the clock starts."""
    ttnn.execute_trace(mesh_device, copy_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start_ns = time.perf_counter_ns()
    controller.replay()  # ends in synchronize_device
    return time.perf_counter_ns() - start_ns


@pytest.mark.skipif(not is_blackhole(), reason="GLM prefill MoE requires Blackhole")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("chunks_per_call", [1, 2, 4], ids=["x1", "x2", "x4"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            mesh_shape,
            make_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config), trace_region_size=_TRACE_REGION_SIZE
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=topology),
            id=mesh_id,
        )
        for mesh_id, mesh_shape, make_device_params, topology in _MESHES
    ],
    indirect=["mesh_device", "device_params"],
)
# The DSv3 variant, as test_glm_moe: it supplies only the generic gate config; the MoE dims are GLM's.
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["ds-ref"])
def test_glm_moe_rows_sweep(variant, config_only, mesh_device, device_params, num_links, chunks_per_call, request):
    """One traced GLM MoE forward at chunks_per_call x 640 rows per chip; logs the replay time."""
    seq_len_per_chip = chunks_per_call * PREFILL_CHUNK_TOKENS_PER_CHIP
    replays_ns = []

    def measure(forward, tt_moe, tt_x):
        controller = SubDeviceTraceController(mesh_device)
        copy_tid = None
        capturing = False
        tt_moe.set_trace_controller(controller)
        try:
            compile_out, _ = forward(ttnn.clone(tt_x))
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(compile_out)

            copy_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            x = ttnn.clone(tt_x)
            ttnn.end_trace_capture(mesh_device, copy_tid, cq_id=0)

            controller.begin_capture()
            capturing = True
            result = forward(x)
            controller.end_capture()
            capturing = False

            for _ in range(max(1, _REPLAYS)):
                replays_ns.append(_replay_ns(mesh_device, copy_tid, controller))
            return result
        finally:
            if capturing:
                controller.end_capture()
            controller.release()
            if copy_tid is not None:
                ttnn.release_trace(mesh_device, copy_tid)
            tt_moe.set_trace_controller(None)
            tt_moe.release_sub_device_manager()

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        seq_len_per_chip,
        GLM53Config.EMB_SIZE,
        GLM53Config.MOE_INTERMEDIATE_SIZE,
        GLM53Config.NUM_ROUTED_EXPERTS,
        GLM53Config.NUM_EXPERTS_PER_TOKEN,
        _DISPATCH_BUFFER_CAPACITY_FACTOR,
        False,  # run_pcc_check: test_glm_moe owns correctness
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        GateComputeMode.DEVICE_FP32,
        request,
        measure=measure,
        overlap_routed_expert_with_combine=routed_expert_combine_overlap_arg(GLM53Config),
    )

    assert replays_ns, "no traced replay was timed"
    median_ms = statistics.median(replays_ns) / 1e6
    mesh = f"{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    logger.warning(
        f"GLM_MOE_SWEEP mesh={mesh} rows_per_chip={seq_len_per_chip} chunks_per_call={chunks_per_call} "
        f"median_ms={median_ms:.3f} per_chunk_ms={median_ms / chunks_per_call:.3f} "
        f"replays_ms={' '.join(f'{ns / 1e6:.3f}' for ns in replays_ns)}"
    )
