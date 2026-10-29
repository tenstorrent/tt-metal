# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TtMoe with moe_transport="fabric2d": traced replays against the same program run eagerly, bit for bit.

Eager, TtMoe runs dispatch_fabric2d, the routed expert, then a standalone combine_fabric2d. With a trace controller
attached it runs the routed expert and combine_fabric2d as one overlapped program, or with
overlap_routed_expert_with_combine=False the same standalone ops. A replay runs every op back to back with no host in
between, so a chip can start the next op while a neighbour is still delivering into its buffers; eager launches are
paced by the host and hide that. Two inputs alternate through one captured input buffer, so a page that had not
landed shows up as the other input's data rather than as the same bytes again, and every replay must match the
eager run of the same program on the same input exactly. Accuracy against torch is test_ttnn_moe.py's job.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatch2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tests.ttnn.utils_for_testing import comp_pcc

_TRACE_REGION_SIZE = 16 * 1024 * 1024
_EAGER_LAUNCHES = 8
# The first replay is where a missing completion wait shows; the rest catch one that only sometimes loses.
_REPLAYS = 10
# The standalone combine and the overlapped program reduce in different orders, so they agree only to rounding.
_OVERLAP_VS_STANDALONE_PCC = 0.999


def _case(variant, model_cfg, dispatch_buffer_capacity_factor, case_id):
    """One model on the 8x4 TorusXY Galaxy, its mesh opened with the fabric payload the model ships with."""
    return pytest.param(
        variant,
        model_cfg,
        dispatch_buffer_capacity_factor,
        (8, 4),
        torus_xy_device_params(fabric_payload_size=moe_fabric_payload(model_cfg), trace_region_size=_TRACE_REGION_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id=f"{case_id}-torus-xy-8x4",
    )


# The fabric2d rows of test_ttnn_moe.py: the two models that opt into the transport.
_CASES = [
    _case("deepseek_v3_d_p", GLM53Config, 8, "glm"),
    _case("kimi_k2_7", KimiK27Config, 5, "kimi_k2_7"),
]


def _host(tensor):
    return [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]


def _assert_same(got, want, what):
    differ = [i for i, (g, w) in enumerate(zip(got, want)) if not torch.equal(g, w)]
    assert not differ, f"{what} differs from its reference on devices {differ}"


def _assert_meaningful(host, what):
    """An output that is all zero, or not finite, would match a wrong one; the comparisons would prove nothing."""
    stacked = torch.stack([t.float() for t in host])
    assert torch.isfinite(stacked).all(), f"{what} has non-finite values"
    assert stacked.abs().sum() > 0, f"{what} is all zero"


@pytest.mark.parametrize("padded_percent", [0, 50], ids=["pad0", "pad50"])
@pytest.mark.parametrize("overlap_routed_expert_with_combine", [None, False], ids=["overlap_auto", "overlap_off"])
@pytest.mark.parametrize(
    "variant, model_cfg, dispatch_buffer_capacity_factor, mesh_device, device_params, num_links",
    _CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1400)
def test_moe_fabric2d_traced_matches_eager(
    variant,
    model_cfg,
    dispatch_buffer_capacity_factor,
    overlap_routed_expert_with_combine,
    padded_percent,
    config_only,
    mesh_device,
    device_params,
    num_links,
    request,
):
    def measure(forward, tt_moe, tt_x):
        # A plumbing slip that left the transport or the overlap at its default would still match itself.
        assert type(tt_moe.dispatch_module) is TtDispatch2dModule, type(tt_moe.dispatch_module)
        assert type(tt_moe.combine_module) is TtCombine2dModule, type(tt_moe.combine_module)
        expected_overlap = overlap_routed_expert_with_combine is not False
        assert tt_moe.overlap_routed_expert_with_combine == expected_overlap
        # Left to the op, dispatch's forwarding buffer is allocated mid-forward once per cached program.
        assert tt_moe.dispatch_module.forwarding_buffer is not None, "TtMoe did not pass the shared dispatch buffers"

        # A second input that routes differently: the gate ranks -x's scores in the opposite order.
        inputs = [tt_x, ttnn.neg(tt_x)]

        # forward() frees its input, hence a clone per launch. No sync until all are queued.
        eager = [forward(ttnn.clone(inputs[i % 2]))[0] for i in range(_EAGER_LAUNCHES)]
        ttnn.synchronize_device(mesh_device)
        eager_host = [_host(out) for out in eager[:2]]
        for launch in range(2, _EAGER_LAUNCHES):
            _assert_same(
                _host(eager[launch]),
                eager_host[launch % 2],
                f"eager launch {launch} of {_EAGER_LAUNCHES} without a host sync",
            )
        for out in eager:
            ttnn.deallocate(out)
        logger.info(f"{_EAGER_LAUNCHES} eager forwards without a host sync are identical per input")

        controller = SubDeviceTraceController(mesh_device)
        copy_tids = []
        capturing = False
        tt_moe.set_trace_controller(controller)
        try:
            # The program a trace replays, run eagerly on each input: the references. A capture cannot JIT-build
            # kernels either, so this also compiles it.
            references = []
            for i, x in enumerate(inputs):
                out, _ = forward(ttnn.clone(x))
                ttnn.synchronize_device(mesh_device)
                references.append(_host(out))
                ttnn.deallocate(out)
                _assert_meaningful(references[i], f"input {i}'s output")
                if expected_overlap:
                    _, pcc = comp_pcc(torch.cat(eager_host[i]).float(), torch.cat(references[i]).float())
                    assert pcc > _OVERLAP_VS_STANDALONE_PCC, f"input {i}: overlapped vs standalone combine PCC {pcc}"
                else:
                    # Without the overlap the controller changes nothing about the program.
                    _assert_same(references[i], eager_host[i], f"input {i} with the controller attached")

            # forward() frees its input mid-capture and the trace reuses that address, so one-op traces copy either
            # input into it before every replay.
            x = ttnn.clone(tt_x)
            for source in inputs:
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                ttnn.copy(source, x)
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                copy_tids.append(tid)

            controller.begin_capture()
            capturing = True
            result = forward(x)
            controller.end_capture()
            capturing = False

            for replay in range(_REPLAYS):
                which = replay % 2
                ttnn.execute_trace(mesh_device, copy_tids[which], cq_id=0, blocking=False)
                controller.replay()  # ends in synchronize_device
                _assert_same(_host(result[0]), references[which], f"traced replay {replay} (input {which})")
            logger.info(f"{_REPLAYS} traced replays, alternating two inputs, match the eager runs")
            return result
        finally:
            if capturing:  # close the open segment so release() runs and the real error surfaces
                controller.end_capture()
            controller.release()
            for tid in copy_tids:
                ttnn.release_trace(mesh_device, tid)
            tt_moe.set_trace_controller(None)
            tt_moe.release_sub_device_manager()

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        model_cfg.EMB_SIZE,
        model_cfg.MOE_INTERMEDIATE_SIZE,
        model_cfg.NUM_ROUTED_EXPERTS,
        model_cfg.NUM_EXPERTS_PER_TOKEN,
        dispatch_buffer_capacity_factor,
        False,  # run_pcc_check: test_ttnn_moe.py owns accuracy
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        GateComputeMode.DEVICE_FP32,
        request,
        padded_percent=padded_percent,
        measure=measure,
        overlap_routed_expert_with_combine=overlap_routed_expert_with_combine,
        moe_transport="fabric2d",
        combine_fwd_reduction=model_cfg.COMBINE_FWD_REDUCTION,
        realistic_weights=True,
        # The model's own dispatch between the routed expert's two kernels; the variant's config may not carry it.
        routed_expert_hybrid_token_threshold=model_cfg.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD,
    )
