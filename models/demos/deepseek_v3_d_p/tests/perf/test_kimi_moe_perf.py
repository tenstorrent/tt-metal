# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi MoE perf gate (K3): host wall-clock of the MoE forward, untraced (fastest of 10 after 10 warm-up,
tests/measure) and traced (one replay), with the CPU clock held; bands untraced ±5%, traced ±3%; the traced output
must match the untraced one.

The forward overlaps dispatch with the shared expert on two sub-devices, so it is captured with
``SubDeviceTraceController`` as several trace segments: an untimed pass compiles, then capture, then one timed
replay. The time covers every segment, the sub-device load/clear between them and the final device sync.

``KIMI_MOE_PERF_REPLAYS=N`` logs N replays; only the first gates. A case whose ``expected_ns`` or
``expected_untraced_ns`` is ``None`` measures that mode without gating. Select with ``-k galaxy`` / ``-k loudbox``;
the LoudBox row (2x4 Fabric2D) is record-only.
"""

import os
import time
from dataclasses import dataclass, field

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.measure.gate import MARGIN, gate
from models.demos.deepseek_v3_d_p.tests.measure.timing import free, time_untraced
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import routed_expert_combine_overlap_arg
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

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
    # Galaxy targets, 3 runs on one Galaxy: traced one replay, untraced fastest of 10 after 10 warm-up.
    expected_ns: int | None
    expected_untraced_ns: int | None
    shape_note: str
    extra: dict = field(default_factory=dict)


_K3 = _MoEPerfCase(
    label="kimi-k3",
    config=KimiK3Config,
    expected_ns=6_036_000,
    expected_untraced_ns=6_392_000,
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

# (variant, case, id).
_CASES = [("kimi_k3", _K3, "k3")]

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
    """Untraced and traced E2E of one MoE forward at the 5k production chunk, per Kimi generation."""
    is_galaxy = tuple(mesh_device.shape) == (8, 4)
    if is_galaxy and not _IGNORE_POWER:
        require_high_power(f"{case.label} MoE perf")
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_tag = "8x4 torus-xy" if is_galaxy else f"{mesh_device.shape[0]}x{mesh_device.shape[1]} fabric2d"

    replays_ns, untraced_ms = [], []
    trace_info = {}
    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)

    def per_device(t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(ttnn.from_device(t))]

    def measure(forward, tt_moe, tt_x):
        # Untraced first. forward() frees its input, so each call gets a clone, inside the timed window (one copy of
        # the chunk's activations; the traced replay's input refresh runs before its clock starts).
        samples, out = time_untraced(lambda: forward(ttnn.clone(tt_x)), lambda: ttnn.synchronize_device(mesh_device))
        untraced_ms.extend(samples)
        eager = per_device(out[0])
        free(out)
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
                if len(replays_ns) == 1:  # the replay's input is tt_x again, as the eager calls' was
                    trace_info["pcc"] = min(comp_pcc(a, b)[1] for a, b in zip(eager, per_device(result[0])))
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
        dispatch_fwd_vc_pages=case.config.MOE_FWD_VC_PAGES,
        combine_fwd_vc_pages=case.config.MOE_FWD_VC_PAGES,
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

    results = {}
    for mode, samples, measured, expected in (
        ("untraced", untraced_ms, min(untraced_ms), case.expected_untraced_ns),
        ("traced", [ns / 1e6 for ns in replays_ns], total_ns / 1e6, case.expected_ns),
    ):
        target = expected / 1e6 if is_galaxy and expected is not None else None  # the LoudBox row records only
        results[mode] = (samples, measured, target, MARGIN[mode])
        emit_perf_metrics(
            f"{case.label.replace('.', '_')}_moe_{mesh_tag.replace(' ', '_')}_{mode}",
            f"{case.label} moe {mesh_tag} {mode} ({case.shape_note}) perf",
            {"mesh": mesh_tag, "mode": mode, "samples_ms": samples, "measured_ms": measured, "target_ms": target},
        )
    gate(f"{case.label} moe {mesh_tag}", results, trace_info["pcc"])
