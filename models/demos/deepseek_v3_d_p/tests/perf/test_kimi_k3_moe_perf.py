# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K3 LatentMoE device-perf gate on the 8x4 galaxy, measured with the real-time program
profiler -- same mechanism as ``test_ttnn_hca_perf.py``, which already gates HCA perf on this
SKU.

Why not the tracy harness the deepseek MoE gates in ``test_moe_perf.py`` use: blaze builds
with the profiler disabled (``blaze-models-prefill-tests.yaml`` passes no ``tracy:`` input ->
``build-artifact.yaml`` adds ``--disable-profiler`` -> ``ENABLE_TRACY=OFF``, which early-returns
before tracy's capture/csvexport targets), so ``build/tools/profiler/bin`` is empty there and the
``python -m tracy`` subprocess that ``run_model_device_perf_test_with_merge`` shells out to has
nothing to run. The real-time profiler is an in-process device API
(``tt_metal/distributed/realtime_profiler_manager.cpp``), not gated on ``TRACY_ENABLE``, so it
measures on the pipeline's ordinary build. It also drops the nested pytest re-run, which is what
keeps this inside the leg's 22-minute slot.

What the number is: over the programs the MoE forward dispatched, the sum of each program's
critical path (max duration across the 32 chips). Close to but NOT the same as the tracy
baseline it replaces -- ``merge_device_rows`` averages CCL ops across devices where this takes
their max -- so this baseline was measured with the real-time profiler and must not be
back-ported to the tracy path. For reference the tracy number was 12_924_852 ns, split Matmul
2_196 us / CCL 1_042 us / Other 9_687 us (dispatch/combine/top-k dominates).

Report-only until calibrated: ``_EXPECTED_NS`` starts as ``None``, so the first galaxy runs
measure and log without gating. See the constant for what to do with the logged number.

Two limits carried over from the tracy version:

  * Measures the SiLU path, not the checkpoint's SiTU-GLU (#51335), so this baseline moves when
    that kernel lands.
  * Forward only. At 896 experts the constructor's one-time weight tilize/typecast is a large
    share of wall time but is not per-token cost; the profiler window is the forward call, so it
    is excluded by construction (no signposts needed).
"""

import os

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

# 640 tokens/chip over SP=8 = the 5k ISL the pcc leg's `kimi_k3-5k-perf` case runs.
_SEQ_LEN_PER_CHIP = 640
# Capacity factor 5 carries over from K2.6, as in the pcc parametrize.
_DISPATCH_BUFFER_CAPACITY_FACTOR = 5

# None = report only, no perf gate. The tracy baseline this replaces (12_924_852 ns) is not a
# comparable number (see the module docstring), so there is nothing honest to assert against until
# the real-time profiler has reported once on the galaxy. Calibration is one step: take the
# "realtime perf" ns this test logs on its first green run and set it here (TODO #53269) -- that
# alone turns the gate on. The measurement itself still fails loudly if it produced no records.
_EXPECTED_NS = None
_MARGIN = 0.03

# The profiler's default 1s collection deadline is sized for a single block's programs. The MoE
# forward at 896 experts dispatches far more, and records arrive asynchronously from the receiver
# thread: a record still in flight when the window closes is NOT counted as dropped, so a short
# deadline would silently under-report. Costs nothing when records arrive promptly -- collection
# still exits on the settle window, not the deadline.
_RECORD_TIMEOUT_S = 5.0

# The team gates perf on the 14kW hosts. Set this to run anywhere for bring-up, where the
# baseline describes nothing and only "does it run" is being checked.
_IGNORE_POWER = os.environ.get("K3_MOE_PERF_IGNORE_POWER") == "1"


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 LatentMoE requires Blackhole")
@pytest.mark.skipif(
    not (is_high_power() or _IGNORE_POWER),
    reason="perf job requires a high-power (>=130W TDP) galaxy; guards the exabox.tenstorrent.com/power=14kw "
    "label. K3_MOE_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k3"], indirect=True, ids=["kimi_k3"])
def test_kimi_k3_moe_perf_galaxy(variant, config_only, mesh_device, device_params, num_links, topology, request):
    """896 experts / top-16, 3584 latent: device time of one MoE forward at 5k ISL."""
    require_realtime_profiler("the Kimi-K3 MoE perf gate")

    per_program = {}

    def measure(forward):
        result, records = profile_realtime_program_merged(
            mesh_device, forward, record_timeout_seconds=_RECORD_TIMEOUT_S
        )
        per_program.update(records)
        return result

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        _SEQ_LEN_PER_CHIP,
        KimiK3Config.EMB_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        KimiK3Config.NUM_ROUTED_EXPERTS,
        KimiK3Config.NUM_EXPERTS_PER_TOKEN,
        _DISPATCH_BUFFER_CAPACITY_FACTOR,
        False,  # run_pcc_check -- the pcc leg (kimi_k3-5k-pcc) owns correctness
        num_links,
        topology,
        GateComputeMode.DEVICE_FP32,
        request,
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        measure=measure,
    )

    # run_model returns early on the perf path, so an empty dict means measure() never ran --
    # i.e. the forward was not the thing profiled. Never report green off that.
    assert per_program, "real-time profiler produced no program records for the MoE forward"

    total_ns = sum(entry["duration_ns"] for entry in per_program.values())

    if _EXPECTED_NS is None:
        logger.warning(
            f"kimi-k3 moe 8x4 realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over "
            f"{len(per_program)} programs -- REPORT ONLY, this run does NOT gate perf. "
            f"Set _EXPECTED_NS = {total_ns:,.0f} in {os.path.basename(__file__)} (TODO #53269) to start gating."
        )
        return

    margin = adjust_margin_for_ddr_speed(_MARGIN)
    lower, upper = _EXPECTED_NS * (1 - margin), _EXPECTED_NS * (1 + margin)
    logger.info(
        f"kimi-k3 moe 8x4 realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over "
        f"{len(per_program)} programs, expected {_EXPECTED_NS:,} ns, "
        f"band [{lower:,.0f}, {upper:,.0f}] (margin +/- {margin * 100:.1f}%)"
    )
    assert lower <= total_ns <= upper, (
        f"device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {_EXPECTED_NS:,} ns, margin +/- {margin * 100:.1f}%)"
    )
