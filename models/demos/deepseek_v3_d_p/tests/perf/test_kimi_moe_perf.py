# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi MoE device-perf gate on the 8x4 galaxy, one test parametrized over the Kimi generations
(K2.6 dense-expert MoE and K3 LatentMoE), measured with the real-time program profiler -- same
mechanism as ``test_ttnn_hca_perf.py``, which already gates HCA perf on this SKU.

What the number is: over the programs the MoE forward dispatched, the sum of each program's
critical path (max duration across the 32 chips). Close to but NOT the same as the tracy
baseline it replaces -- ``merge_device_rows`` averages CCL ops across devices where this takes
their max -- so these baselines were measured with the real-time profiler and must not be
back-ported to the tracy path.

What the number excludes, verified on an 8x4 galaxy (warm caches, 58 programs x 32 chips =
1856 records, 0 dropped, no program dispatched more than once per chip):

  * **Op-to-op dispatch gaps: 7.29 ms, excluded.** A record runs from the dispatch_s loop-top that
    picks up the program's GO command to the last worker's completion semaphore
    (``cq_dispatch_subordinate.cpp``, ``record_realtime_timestamp`` at both ends), so the time from
    one program's workers-done to the next program's loop-top falls *between* records. The forward's
    first_start->last_end span was 19.39 ms against a 12.53 ms sum of program durations.
  * **Host stalls: excluded, but only because the measured pass is warm.** The start stamp is taken
    before ``cb_acquire_pages_dispatch_s`` blocks on host-fed pages, so when the host lags the gaps
    get pulled *inside* the records: the same forward measured cold (JIT compiling ~1080 kernels)
    reported 19.01 ms, i.e. the E2E span rather than the sum. Hence the warm-up pass in the test.

Recalibrating one generation: set its ``expected_ns`` to ``None`` and the test measures and logs
without gating, printing the value to set it back to. Do that on any box whose baseline you need to
re-cut. One shared codepath, one job per generation: ``kimi_moe_perf`` runs ``-k k2_6`` and
``kimi_k3_moe_perf`` runs ``-k k3``, so a regression in one generation cannot mask or block the
other, and each carries its own timeout.
"""

import os
from dataclasses import dataclass, field

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

# 640 tokens/chip over SP=8 = the 5120-token chunk production prefill feeds, and the same shape the
# pcc legs (`kimi-5k-pcc`, `kimi_k3-5k-pcc`) run.
_SEQ_LEN_PER_CHIP = 640
# Capacity factor 5 for both: K3 halves the row width (7168 -> 3584 latent) and doubles the token
# slots (top-8 -> top-16), so per-chip dispatch bytes are roughly unchanged.
_DISPATCH_BUFFER_CAPACITY_FACTOR = 5

# Both generations carry FABRIC_PAYLOAD_SIZE = 7168, so one device_params axis serves both.
_FABRIC_PAYLOAD_SIZE = KimiK26Config.FABRIC_PAYLOAD_SIZE

# The profiler's default 1s collection deadline is sized for a single block's programs. The MoE
# forward at 896 experts dispatches far more, and records arrive asynchronously from the receiver
# thread: a record still in flight when the window closes is NOT counted as dropped, so a short
# deadline would silently under-report. Costs nothing when records arrive promptly -- collection
# still exits on the settle window, not the deadline.
_RECORD_TIMEOUT_S = 5.0

# The team gates perf on the 14kW hosts. Set this to run anywhere for bring-up, where the
# baseline describes nothing and only "does it run" is being checked.
_IGNORE_POWER = os.environ.get("KIMI_MOE_PERF_IGNORE_POWER") == "1"


@dataclass(frozen=True)
class _MoEPerfCase:
    """One generation's shape, baseline, and the run_model kwargs unique to it."""

    label: str
    config: type
    expected_ns: int | None
    margin: float
    shape_note: str
    extra: dict = field(default_factory=dict)


# K2.6: 384 experts / top-8 over the 7168 embedding, no LatentMoE plumbing.
# TODO: cut on a high-power 8x4 galaxy and set expected_ns to start gating. Until then this case
# measures and reports without asserting, so it cannot go green off an unmeasured baseline.
_K2_6 = _MoEPerfCase(
    label="kimi-k2.6",
    config=KimiK26Config,
    expected_ns=None,
    # Placeholder, carried from K3 pending this case's own observed spread. K3's 3% was justified by
    # repeated warm runs on that box; re-check against K2.6's own numbers when the baseline is cut.
    margin=0.03,
    shape_note="384 experts / top-8, 7168 emb",
)

# K3: 896 experts / top-16, 3584 latent.
#
# Re-centered 2026-08-25 (issue #54280): the forward got ~4.6% FASTER and fell out the bottom of the
# band, so the baseline was stale rather than the margin too tight. Likely source is #53968
# (active-ERISC __global_pointer$ link fix, 2026-08-21 16:19 UTC) -- same suspect and direction as
# #54220 on the Kimi-K2.6 traced chunked gate; the previous 12,210,765 was measured 2026-08-21,
# plausibly just before it landed.
#
# Measured on an 8x4 BH galaxy (nominal DDR, high power), warm forward, routed experts folded into
# one program: run 32811686276/job 97720001496 (main). Run 32728173507 independently measured
# 11,555,528 ns at the same 31-program shape, 0.8% under this value and well inside the band.
#
# One carried-over limit: this value was cut before the shared expert moved off SiLU. It now measures
# the checkpoint's SiTU-GLU on every FFN site, where the shared expert's single fused multiply
# becomes a softcap/sigmoid/multiply chain, so the baseline is stale until re-cut.
_K3 = _MoEPerfCase(
    label="kimi-k3",
    config=KimiK3Config,
    expected_ns=11_646_483,
    # Repeated warm measurements on that box spanned 0.63% stdev / 1.89% peak to peak, so 3% holds
    # the observed run-to-run noise; sub-nominal DDR doubles it to 6% via adjust_margin_for_ddr_speed.
    # The baseline above is a single run, not the centre of a spread -- recentre it if a regression
    # this shallow ever needs catching.
    margin=0.03,
    shape_note="896 experts / top-16, 3584 latent",
    extra=dict(
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        shared_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
    ),
)

# "k2_6" / "k3", not "kimi_k2_6" / "kimi_k3": pytest -k is substring-based, so the ids must stay
# disjoint -- a bare "kimi" id would match both generations and widen every `-k` selector.
_CASES = [
    pytest.param("kimi_k2_6", _K2_6, id="k2_6"),
    pytest.param("kimi_k3", _K3, id="k3"),
]


@pytest.mark.skipif(not is_blackhole(), reason="Kimi prefill MoE requires Blackhole")
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
            torus_xy_device_params(fabric_payload_size=_FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant, case", _CASES, indirect=["variant"])
def test_kimi_moe_perf_galaxy(variant, case, config_only, mesh_device, device_params, num_links, request):
    """Device time of one MoE forward at the 5k production chunk, per Kimi generation."""
    require_realtime_profiler(f"the {case.label} MoE perf gate")
    topology = per_axis_topology(device_params["fabric_config"])

    per_program = {}

    def measure(forward):
        # Warm-up pass, discarded. An RT record starts when the dispatcher picks up the command --
        # before cb_acquire_pages_dispatch_s blocks on host-fed pages -- so host stalls land inside
        # the measured window. On a cold run that means JIT compilation of ~1080 kernels. Dispatch
        # overhead itself is device work and stays in the number; host compile time is not device
        # time and must not be.
        warm = forward()
        ttnn.synchronize_device(mesh_device)
        del warm  # up to 896 experts: free the discarded outputs before the measured pass

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
    )

    # run_model returns early on the perf path, so an empty dict means measure() never ran --
    # i.e. the forward was not the thing profiled. Never report green off that.
    assert per_program, f"real-time profiler produced no program records for the {case.label} MoE forward"

    total_ns = sum(entry["duration_ns"] for entry in per_program.values())
    tag = f"{case.label} moe 8x4 realtime perf ({case.shape_note})"

    if case.expected_ns is None:
        logger.warning(
            f"{tag}: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over {len(per_program)} programs "
            f"-- REPORT ONLY, this run does NOT gate perf. Set expected_ns = {total_ns:,.0f} for "
            f"{case.label} in {os.path.basename(__file__)} to start gating."
        )
        return

    margin = adjust_margin_for_ddr_speed(case.margin)
    lower, upper = case.expected_ns * (1 - margin), case.expected_ns * (1 + margin)
    logger.info(
        f"{tag}: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms) over {len(per_program)} programs, "
        f"expected {case.expected_ns:,} ns, band [{lower:,.0f}, {upper:,.0f}] "
        f"(margin +/- {margin * 100:.1f}%)"
    )
    assert lower <= total_ns <= upper, (
        f"{case.label}: device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {case.expected_ns:,} ns, margin +/- {margin * 100:.1f}%)"
    )
