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
import time
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
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import routed_expert_combine_overlap_arg
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

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
    expected_ns=3_103_000,
    margin=0.03,
    shape_note="384 experts / top-8, 7168 emb",
)

_K3 = _MoEPerfCase(
    label="kimi-k3",
    config=KimiK3Config,
    # K3 does not opt into the routed expert / combine overlap; 6.31 ms is its Oct 16 baseline, from before
    # the overlap existed (#60928's runs: 6.304 / 6.316 / 6.308 ms).
    expected_ns=6_310_000,
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


def _replay_ns(mesh_device, copy_tid, controller):
    """Wall-clock of one segmented replay; the input copy runs before the clock starts."""
    ttnn.execute_trace(mesh_device, copy_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start_ns = time.perf_counter_ns()
    controller.replay()  # ends in synchronize_device
    return time.perf_counter_ns() - start_ns


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

    replays_ns = []
    trace_info = {}

    def measure(forward, tt_moe, tt_x):
        controller = SubDeviceTraceController(mesh_device)
        copy_tid = None
        capturing = False
        tt_moe.set_trace_controller(controller)
        try:
            # Compile pass (a capture cannot JIT-build kernels). forward() frees its input, hence clone.
            compile_out, _ = forward(ttnn.clone(tt_x))
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(compile_out)

            # forward() frees its input mid-capture, so its address gets reused inside the trace. A
            # separate one-op trace refreshes that buffer from tt_x before every replay, keeping the
            # routed input identical across replays.
            copy_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            x = ttnn.clone(tt_x)
            ttnn.end_trace_capture(mesh_device, copy_tid, cq_id=0)

            controller.begin_capture()
            capturing = True
            result = forward(x)
            controller.end_capture()
            capturing = False
            trace_info["segments"] = controller.num_segments
            trace_info["bytes"] = controller.trace_bytes()

            for _ in range(max(1, _REPLAYS)):
                replays_ns.append(_replay_ns(mesh_device, copy_tid, controller))
            return result
        finally:
            if capturing:  # close the open segment so release() runs and the real error surfaces
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
    )

    assert replays_ns, f"no traced replay was timed for the {case.label} MoE forward"

    total_ns = replays_ns[0]
    tag = f"{case.label} moe {mesh_tag} traced replay ({case.shape_note})"
    logger.info(
        f"{tag}: {trace_info['segments']} trace segments, {trace_info['bytes'] / 2**20:.2f} MiB trace region "
        f"of {_TRACE_REGION_SIZE / 2**20:.0f} MiB"
    )
    if len(replays_ns) > 1:
        logger.info(f"{tag}: replays (ms) " + " ".join(f"{ns / 1e6:.3f}" for ns in replays_ns))

    expected_ns = case.expected_ns if is_galaxy else None
    margin = None if expected_ns is None else adjust_margin_for_ddr_speed(case.margin)
    emit_perf_metrics(
        f"{case.label.replace('.', '_')}_moe_{mesh_tag.replace(' ', '_')}",
        f"{tag} perf",
        {
            "mesh": mesh_tag,
            "measured_ns": total_ns,
            "measured_ms": total_ns / 1e6,
            "replays_ns": replays_ns,
            "expected_ns": expected_ns,
            "perf_margin_pct": None if margin is None else margin * 100.0,
            "within_band": None
            if expected_ns is None
            else expected_ns * (1 - margin) <= total_ns <= expected_ns * (1 + margin),
        },
    )
    if expected_ns is None:
        logger.warning(f"{tag}: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) -- REPORT ONLY, not gated")
        return

    lower, upper = expected_ns * (1 - margin), expected_ns * (1 + margin)
    logger.info(
        f"{tag}: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms), expected {expected_ns:,} ns, "
        f"band [{lower:,.0f}, {upper:,.0f}] (margin +/- {margin * 100:.1f}%)"
    )
    assert lower <= total_ns <= upper, (
        f"{case.label}: traced replay {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {expected_ns:,} ns, margin +/- {margin * 100:.1f}%)"
    )
