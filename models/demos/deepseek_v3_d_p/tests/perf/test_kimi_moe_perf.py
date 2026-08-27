# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K2.6 MoE device-perf gate on the 8x4 galaxy, measured with the real-time program profiler.
Sibling of ``test_kimi_k3_moe_perf.py``; see that file for the full account of what the number
does and does not include. In short:

What the number is: over the programs the MoE forward dispatched, the sum of each program's
critical path (max duration across the 32 chips). It is close to but NOT the same as a tracy
baseline -- ``merge_device_rows`` averages CCL ops across devices where this takes their max --
so this baseline must not be back-ported to the tracy path.

Excluded: op-to-op dispatch gaps (they fall between records) and host stalls, the latter only
because the measured pass is warm -- hence the warm-up forward below. A cold pass pulls JIT
compilation inside the records and reports the E2E span instead of the sum.

Why this exists: K2.6 had no op-level MoE perf gate. ``test_moe_perf.py`` covers DeepSeek only
(``perf-device-256`` / ``perf-host-64``) and ``test_kimi_k3_moe_perf.py`` covers K3 only, so K2.6
throughput was gated at the model level alone (the ``kimi_chunked`` perf legs). The ``kimi-5k-perf``
row in ``test_ttnn_moe.py`` was selected by no CI yaml and gated nothing.

Recalibrating: set ``_EXPECTED_NS = None`` and the test measures and logs without gating, printing
the value to set it back to. Do that on any box whose baseline you need to re-cut.
"""

import os

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

# 640 tokens/chip over SP=8 = the 5120-token chunk production prefill feeds, and the same shape the
# pcc leg's `kimi-5k-pcc` case runs.
_SEQ_LEN_PER_CHIP = 640
# Capacity factor 5, as in the pcc parametrize.
_DISPATCH_BUFFER_CAPACITY_FACTOR = 5

# TODO: cut on a high-power 8x4 galaxy and set the measured value here to start gating. Until then
# the test runs and reports without asserting, so it cannot go green off an unmeasured baseline.
_EXPECTED_NS = None
# Matches the K3 gate: repeated warm measurements there spanned 0.63% stdev / 1.89% peak to peak, so
# 3% holds run-to-run noise; sub-nominal DDR doubles it to 6% via adjust_margin_for_ddr_speed.
# Re-check against this suite's own observed spread when the baseline is cut.
_MARGIN = 0.03

# The profiler's default 1s collection deadline is sized for a single block's programs. Records
# arrive asynchronously from the receiver thread and one still in flight when the window closes is
# NOT counted as dropped, so a short deadline would silently under-report. Costs nothing when
# records arrive promptly -- collection still exits on the settle window, not the deadline.
_RECORD_TIMEOUT_S = 5.0

# The team gates perf on the 14kW hosts. Set this to run anywhere for bring-up, where the baseline
# describes nothing and only "does it run" is being checked.
_IGNORE_POWER = os.environ.get("KIMI_MOE_PERF_IGNORE_POWER") == "1"


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K2.6 prefill MoE requires Blackhole")
@pytest.mark.skipif(
    not (is_high_power() or _IGNORE_POWER),
    reason="perf job requires a high-power (>=130W TDP) galaxy; guards the exabox.tenstorrent.com/power=14kw "
    "label. KIMI_MOE_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi_k2_6"])
def test_kimi_moe_perf_galaxy(variant, config_only, mesh_device, device_params, num_links, request):
    """384 experts / top-8, 7168 emb: device time of one MoE forward at the 5k production chunk."""
    require_realtime_profiler("the Kimi-K2.6 MoE perf gate")
    topology = per_axis_topology(device_params["fabric_config"])

    per_program = {}

    def measure(forward):
        # Warm-up pass, discarded. An RT record starts when the dispatcher picks up the command --
        # before cb_acquire_pages_dispatch_s blocks on host-fed pages -- so host stalls land inside
        # the measured window, and on a cold run that means JIT compilation. Dispatch overhead
        # itself is device work and stays in the number; host compile time is not device time.
        warm = forward()
        ttnn.synchronize_device(mesh_device)
        del warm  # free the discarded outputs before allocating the measured pass

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
        KimiK26Config.EMB_SIZE,
        KimiK26Config.MOE_INTERMEDIATE_SIZE,
        KimiK26Config.NUM_ROUTED_EXPERTS,
        KimiK26Config.NUM_EXPERTS_PER_TOKEN,
        _DISPATCH_BUFFER_CAPACITY_FACTOR,
        False,  # run_pcc_check -- the pcc leg (kimi-5k-pcc) owns correctness
        num_links,
        topology,
        GateComputeMode.DEVICE_FP32,
        request,
        measure=measure,
    )

    # run_model returns early on the perf path, so an empty dict means measure() never ran --
    # i.e. the forward was not the thing profiled. Never report green off that.
    assert per_program, "real-time profiler produced no program records for the MoE forward"

    total_ns = sum(entry["duration_ns"] for entry in per_program.values())

    if _EXPECTED_NS is None:
        logger.warning(
            f"kimi-k2.6 moe 8x4 realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over "
            f"{len(per_program)} programs -- REPORT ONLY, this run does NOT gate perf. "
            f"Set _EXPECTED_NS = {total_ns:,.0f} in {os.path.basename(__file__)} to start gating."
        )
        return

    margin = adjust_margin_for_ddr_speed(_MARGIN)
    lower, upper = _EXPECTED_NS * (1 - margin), _EXPECTED_NS * (1 + margin)
    logger.info(
        f"kimi-k2.6 moe 8x4 realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over "
        f"{len(per_program)} programs, expected {_EXPECTED_NS:,} ns, "
        f"band [{lower:,.0f}, {upper:,.0f}] (margin +/- {margin * 100:.1f}%)"
    )
    assert lower <= total_ns <= upper, (
        f"device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {_EXPECTED_NS:,} ns, margin +/- {margin * 100:.1f}%)"
    )
