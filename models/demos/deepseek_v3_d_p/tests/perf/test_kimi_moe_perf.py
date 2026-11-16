# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi MoE perf gate (K2.7 and K3): host wall-clock of ONE traced replay of the MoE forward.

The forward overlaps dispatch with the shared expert on two sub-devices, so it is captured with
``SubDeviceTraceController`` as several trace segments. Sequence: an untimed eager pass to compile
the kernels, capture, then one timed replay -- no warm-up replay, no averaging. The time covers every
segment, the sub-device load/clear between them and the final device sync.

``KIMI_MOE_PERF_REPLAYS=N`` logs N replays; only the first gates. Setting a case's ``expected_ns`` to
``None`` measures without gating. Select with ``-k k2_7`` / ``-k k3`` and ``-k galaxy`` / ``-k loudbox``;
the LoudBox row (2x4 Fabric2D) is record-only.
"""

import os
from dataclasses import dataclass, field

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.measure.gate import PerfReport, PerfTarget
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import routed_expert_combine_overlap_arg
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power

# 640 tokens/chip: the 5k production prefill chunk on SP=8.
_SEQ_LEN_PER_CHIP = PREFILL_CHUNK_TOKENS_PER_CHIP
# K3 halves the routed row width and doubles top-k, so per-chip dispatch bytes match K2.7.
_DISPATCH_BUFFER_CAPACITY_FACTOR = 5
# One forward uses ~0.26 MiB per chip (3 segments); the test logs the actual size.
_TRACE_REGION_SIZE = 8 * 1024 * 1024

_REPLAYS = int(os.environ.get("KIMI_MOE_PERF_REPLAYS", "1"))
# Baselines are cut on the 14kW galaxies; this runs elsewhere for bring-up only.
_IGNORE_POWER = os.environ.get("KIMI_MOE_PERF_IGNORE_POWER") == "1"


@dataclass(frozen=True)
class _MoEPerfCase:
    """One generation's shape, Galaxy baseline, and the run_model kwargs unique to it."""

    label: str
    config: type
    expected_ns: int | None
    margin: float
    shape_note: str
    extra: dict = field(default_factory=dict)


_K2_7 = _MoEPerfCase(
    label="kimi-k2.7",
    config=KimiK27Config,
    expected_ns=2_896_000,
    margin=0.03,
    shape_note="384 experts / top-8, 7168 emb",
)

_K3 = _MoEPerfCase(
    label="kimi-k3",
    config=KimiK3Config,
    expected_ns=6_030_000,
    margin=0.03,
    shape_note="896 experts / top-16, 3584 latent",
    extra=dict(
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        routed_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[KimiK3Config.ROUTED_EXPERT_ACTIVATION],
        shared_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
    ),
)

# (variant, case, id). Disjoint ids: pytest -k matches substrings, so a bare "kimi" would select both.
_CASES = [("kimi_k2_7", _K2_7, "k2_7"), ("kimi_k3", _K3, "k3")]

# (id, mesh shape, fabric profile, topology mark). Only the galaxy row gates; the LoudBox row records.
_MESHES = [
    ("galaxy-torus-xy-8x4", (8, 4), torus_xy_device_params, "mesh-8x4"),
    ("loudbox-fabric2d-2x4", (2, 4), fabric2d_device_params, "mesh-2x4"),
]


def _perf_params():
    """Every generation on every mesh, each opening the fabric with its own config's payload size."""
    return [
        pytest.param(
            variant,
            case,
            mesh_shape,
            make_device_params(
                fabric_payload_size=moe_fabric_payload(case.config), trace_region_size=_TRACE_REGION_SIZE
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=topology),
            id=f"{case_id}-{mesh_id}",
        )
        for variant, case, case_id in _CASES
        for mesh_id, mesh_shape, make_device_params, topology in _MESHES
    ]


@pytest.mark.skipif(not is_blackhole(), reason="Kimi prefill MoE requires Blackhole")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "variant, case, mesh_device, device_params, num_links",
    _perf_params(),
    indirect=["variant", "mesh_device", "device_params"],
)
def test_kimi_moe_perf(variant, case, config_only, mesh_device, device_params, num_links, request):
    """Traced E2E of one MoE forward at the 5k production chunk, per Kimi generation."""
    is_galaxy = tuple(mesh_device.shape) == (8, 4)
    if is_galaxy and not _IGNORE_POWER:
        require_high_power(f"{case.label} MoE perf")
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_tag = "8x4 torus-xy" if is_galaxy else f"{mesh_device.shape[0]}x{mesh_device.shape[1]} fabric2d"

    replays = None
    trace_info = {}

    def measure(forward, tt_moe, tt_x):
        nonlocal replays
        state = {}

        def step():
            # forward() frees its input: the compile pass gets a clone, the capture the copy trace's buffer.
            return forward(state.pop("x") if "x" in state else ttnn.clone(tt_x))

        def capture_input_refresh(_compile_out):
            # forward() frees its input mid-capture, so its address gets reused inside the trace. A separate one-op
            # trace refreshes that buffer from tt_x before every replay, keeping the routed input identical.
            ttnn.synchronize_device(mesh_device)
            state["copy_tid"] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            state["x"] = ttnn.clone(tt_x)
            ttnn.end_trace_capture(mesh_device, state["copy_tid"], cq_id=0)

        def detach():
            if "copy_tid" in state:
                ttnn.release_trace(mesh_device, state["copy_tid"])
            tt_moe.set_trace_controller(None)
            tt_moe.release_sub_device_manager()

        with TracedStep(mesh_device, step, attach=tt_moe.set_trace_controller, detach=detach) as traced:
            traced.capture(before_capture=capture_input_refresh)
            trace_info["segments"] = traced.controller.num_segments
            trace_info["bytes"] = traced.controller.trace_bytes()
            replays = time_traced(
                traced,
                samples=max(1, _REPLAYS),
                estimator="first",
                prepare=lambda: ttnn.execute_trace(mesh_device, state["copy_tid"], cq_id=0, blocking=False),
            )
            return traced.take_output()  # run_model owns the result

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        _SEQ_LEN_PER_CHIP,
        case.config.EMB_SIZE,
        case.config.MOE_INTERMEDIATE_SIZE,
        case.config.NUM_ROUTED_EXPERTS,
        case.config.NUM_EXPERTS_PER_TOKEN,
        _DISPATCH_BUFFER_CAPACITY_FACTOR,
        False,  # run_pcc_check -- the pcc legs own correctness
        num_links,
        topology,
        GateComputeMode.DEVICE_FP32,
        request,
        **case.extra,
        measure=measure,
        overlap_routed_expert_with_combine=routed_expert_combine_overlap_arg(case.config),
        dispatch_fwd_vc_pages=case.config.MOE_FWD_VC_PAGES,
        combine_fwd_vc_pages=case.config.MOE_FWD_VC_PAGES,
    )

    assert replays is not None, f"no traced replay was timed for the {case.label} MoE forward"

    tag = f"{case.label} moe {mesh_tag} traced replay ({case.shape_note})"
    logger.info(
        f"{tag}: {trace_info['segments']} trace segments, {trace_info['bytes'] / 2**20:.2f} MiB trace region "
        f"of {_TRACE_REGION_SIZE / 2**20:.0f} MiB"
    )
    expected_ns = case.expected_ns if is_galaxy else None
    margin = adjust_margin_for_ddr_speed(case.margin) if expected_ns is not None else case.margin

    def emit(result):
        replays_ns = [ms * 1e6 for ms in result.samples.values]
        emit_perf_metrics(
            f"{case.label.replace('.', '_')}_moe_{mesh_tag.replace(' ', '_')}",
            f"{tag} perf",
            {
                "mesh": mesh_tag,
                "measured_ns": result.measured * 1e6,
                "measured_ms": result.measured,
                "replays_ns": replays_ns,
                "expected_ns": expected_ns,
                "perf_margin_pct": None if expected_ns is None else margin * 100.0,
                "within_band": None if expected_ns is None else result.within,
            },
        )

    report = PerfReport(tag, emit)
    report.add("traced", replays, PerfTarget(None if expected_ns is None else expected_ns / 1e6, margin))
    if expected_ns is None:
        logger.warning(f"{tag}: {replays.estimate:.3f} ms -- REPORT ONLY, not gated")
    report.finish(skip_unmeasured=False)
