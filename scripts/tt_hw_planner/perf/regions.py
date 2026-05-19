# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Classify each joined row into a labeled performance region.

Regions (matched to the labels described in the report):

  A  saturated at LoFi compute peak     -> nothing to do
  B  saturated at HiFi2 compute peak    -> try math_fidelity_downcast
  C  saturated at HiFi3/HiFi4 peak      -> downcast OR tune program_config
  D  on the bandwidth diagonal          -> balanced; only via fusion / IA up
  E  bandwidth-bound (DRAM/NoC/ETH)     -> dram_l1_promoter, layout_unifier
  F  far below any ceiling              -> trace_capturer, cache_warmer
  ?  insufficient data                  -> not classified

Each row gets `region` + `region_reason` filled in. The chart layer paints
points by region; the optimizer-block sidebar derives suggestions from
which region a cluster ends up in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ceilings import BoxSpec, MATH_FIDELITY_LEVELS
from .join import JoinedRow


# Thresholds. These are deliberately conservative; the report layer can
# expose them as sliders later. Each is justified inline.

# Above this fraction of the FPU ceiling for the *current* fidelity, the op
# is considered saturated. We use the *current* fidelity peak (not the
# global LoFi peak) so HiFi4 ops legitimately land in region C.
SATURATION_FRAC = 0.85

# Above this fraction of the matching BW ceiling, the op is BW-bound.
BW_BOUND_FRAC = 0.70

# Below this fraction of any ceiling, the op is far-below-everything.
HOST_BOUND_FRAC = 0.10

# Op-to-op latency over kernel time => dispatch-bound (region F).
DISPATCH_BOUND_RATIO = 0.5


@dataclass
class Region:
    label: str
    name: str
    suggestions: Tuple[str, ...]
    description: str


REGIONS: Dict[str, Region] = {
    "A": Region(
        label="A",
        name="LoFi-saturated",
        suggestions=(),
        description="At LoFi compute ceiling; effectively maxed out. Nothing further to do.",
    ),
    "B": Region(
        label="B",
        name="HiFi2-saturated",
        suggestions=("math_fidelity_downcast",),
        description="At HiFi2 compute ceiling; downcast to LoFi if accuracy allows.",
    ),
    "C": Region(
        label="C",
        name="HiFi4-saturated",
        suggestions=("math_fidelity_downcast", "program_config_tuner"),
        description="At HiFi3/HiFi4 ceiling; downcast or tune the program_config.",
    ),
    "D": Region(
        label="D",
        name="Balanced",
        suggestions=("fusion_rewriter",),
        description="On the BW diagonal at balanced intensity; improve via fusion or IA increase.",
    ),
    "E": Region(
        label="E",
        name="BW-bound",
        suggestions=("dram_l1_promoter", "fusion_rewriter", "layout_unifier"),
        description="Bandwidth-bound; promote DRAM to L1, fuse adjacent ops, or unify layout.",
    ),
    "F": Region(
        label="F",
        name="Dispatch/host-bound",
        suggestions=("trace_capturer", "cache_warmer"),
        description="Far below all ceilings; capture trace and warm the program cache.",
    ),
    "?": Region(
        label="?",
        name="Insufficient data",
        suggestions=(),
        description="No PM_IDEAL or kernel duration present; cannot classify.",
    ),
}


def _fidelity_peak_flops(row: JoinedRow, box: BoxSpec) -> Tuple[Optional[float], str]:
    """Pick the FPU peak to compare against based on the row's math fidelity."""
    fidelity = (row.math_fidelity or "").strip()
    if fidelity in box.fpu_peak_tflops:
        return box.fpu_peak_flops_per_chip(fidelity) * box.total_chips, fidelity
    return box.fpu_peak_flops_per_chip("HiFi2") * box.total_chips, "HiFi2"


def _bw_ceiling_for(util_pct: Optional[float], bw_bytes_per_s: float) -> Optional[float]:
    """How many bytes/s the op was actually moving; None when util missing."""
    if util_pct is None or bw_bytes_per_s <= 0:
        return None
    return (util_pct / 100.0) * bw_bytes_per_s


def classify_row(row: JoinedRow, box: BoxSpec) -> Tuple[str, str]:
    """Return (region_label, human-readable reason)."""
    if row.device_kernel_ns is None or row.device_kernel_ns <= 0:
        return "?", "no DEVICE KERNEL DURATION"

    # Dispatch-bound check first: a hot op with a large op-to-op gap and
    # almost no compute/BW utilization is region F regardless of intensity.
    if (
        row.op_to_op_latency_ns is not None
        and row.device_kernel_ns > 0
        and row.op_to_op_latency_ns >= DISPATCH_BOUND_RATIO * row.device_kernel_ns
        and (row.pm_fpu_util_pct or 0) < HOST_BOUND_FRAC * 100
        and (row.dram_bw_util_pct or 0) < HOST_BOUND_FRAC * 100
    ):
        return "F", (
            f"op-to-op latency {row.op_to_op_latency_ns:.0f} ns ≥ "
            f"{DISPATCH_BOUND_RATIO:.0%} of kernel {row.device_kernel_ns:.0f} ns "
            f"with low utilization (dispatch-bound)"
        )

    # FPU-saturated checks: pick the fidelity-specific peak.
    fpu_util = row.pm_fpu_util_pct
    fidelity = (row.math_fidelity or "").strip()
    if fpu_util is not None and fpu_util >= SATURATION_FRAC * 100:
        if fidelity == "LoFi":
            return "A", f"FPU util {fpu_util:.1f}% at LoFi (saturated)"
        if fidelity == "HiFi2":
            return "B", f"FPU util {fpu_util:.1f}% at HiFi2"
        return "C", f"FPU util {fpu_util:.1f}% at {fidelity or 'HiFi3/4'}"

    # Bandwidth-bound checks: any of DRAM/NoC/ETH above threshold.
    dram = row.dram_bw_util_pct or 0
    noc = row.noc_util_pct or 0
    eth = row.eth_bw_util_pct or 0
    if dram >= BW_BOUND_FRAC * 100:
        return "E", f"DRAM BW util {dram:.1f}% (bandwidth-bound)"
    if noc >= BW_BOUND_FRAC * 100:
        return "E", f"NoC util {noc:.1f}% (bandwidth-bound)"
    if eth >= BW_BOUND_FRAC * 100:
        return "E", f"ETH BW util {eth:.1f}% (bandwidth-bound)"

    # If we know PM_IDEAL, %-of-peak tells us how close to the BW
    # diagonal we are. Above ~50% with no FPU saturation is "balanced."
    if row.pm_ideal_ns is not None and row.pm_ideal_ns > 0:
        pct_peak = 100.0 * row.pm_ideal_ns / row.device_kernel_ns
        if pct_peak >= 50:
            return "D", f"PM_IDEAL/actual = {pct_peak:.1f}% (balanced on BW diagonal)"

    # Nothing significant. Region F (under-launched / host-bound) is the
    # final catch.
    return "F", (f"low utilization: FPU={fpu_util or 0:.1f}% DRAM={dram:.1f}% NoC={noc:.1f}% " "(under-launched)")


def classify_all(rows: List[JoinedRow], box: BoxSpec) -> None:
    """Mutate each row's `region` and `region_reason`."""
    for r in rows:
        label, reason = classify_row(r, box)
        r.region = label
        r.region_reason = reason


def region_color() -> Dict[str, str]:
    """Hex colors for chart overlay regions. Picked to be readable on dark."""
    return {
        "A": "#2EA043",  # green: saturated, nothing to do
        "B": "#FFD93D",  # yellow: HiFi2-bound
        "C": "#FFA630",  # orange: HiFi4-bound
        "D": "#1F6FEB",  # blue: balanced
        "E": "#F85149",  # red: BW-bound
        "F": "#8B949E",  # gray: dispatch / host-bound
        "?": "#586069",  # dim gray: insufficient data
    }


# ---------------------------------------------------------------------------
# Self-test (`python -m scripts.tt_hw_planner.perf.regions`)
# ---------------------------------------------------------------------------


def _synth_row(**kwargs) -> JoinedRow:
    base = dict(
        run_id="test",
        row_index=0,
        global_call_count=0,
        block_path="root",
        op_code="ttnn.matmul",
        op_type="tt_dnn_device",
        tracer_op_name=None,
        args_hash=None,
        arguments={},
        math_fidelity="HiFi2",
        inputs_summary="",
        outputs_summary="",
        inputs=[],
        outputs=[],
        compute_kernel_source="",
        compute_kernel_hash="",
        dm_kernel_source="",
        dm_kernel_hash="",
        program_hash="",
        program_cache_hit=True,
        core_count=64,
        device_kernel_ns=1000.0,
        device_fw_ns=1200.0,
        op_to_op_latency_ns=50.0,
        brisc_ns=0,
        ncrisc_ns=0,
        trisc0_ns=0,
        trisc1_ns=0,
        trisc2_ns=0,
        erisc_ns=0,
        compute_cb_wait_front_ns=0,
        pm_ideal_ns=500.0,
        pm_compute_ns=500.0,
        pm_bandwidth_ns=400.0,
        pm_req_i_bw=0,
        pm_req_o_bw=0,
        pm_fpu_util_pct=10.0,
        noc_util_pct=10.0,
        multicast_noc_util_pct=0,
        dram_bw_util_pct=10.0,
        eth_bw_util_pct=0.0,
    )
    base.update(kwargs)
    return JoinedRow(**base)


def _self_test() -> None:
    from .ceilings import load_box_spec

    box = load_box_spec("QB2", (1, 4))
    cases = [
        ("A", _synth_row(math_fidelity="LoFi", pm_fpu_util_pct=92.0)),
        ("B", _synth_row(math_fidelity="HiFi2", pm_fpu_util_pct=92.0)),
        ("C", _synth_row(math_fidelity="HiFi4", pm_fpu_util_pct=92.0)),
        ("D", _synth_row(pm_ideal_ns=700.0, device_kernel_ns=1000.0, pm_fpu_util_pct=30.0)),
        ("E", _synth_row(dram_bw_util_pct=80.0)),
        (
            "F",
            _synth_row(
                op_to_op_latency_ns=10_000.0, device_kernel_ns=1000.0, pm_fpu_util_pct=2.0, dram_bw_util_pct=2.0
            ),
        ),
        ("?", _synth_row(device_kernel_ns=None)),
    ]
    for expected, row in cases:
        got, reason = classify_row(row, box)
        status = "PASS" if got == expected else "FAIL"
        print(f"{status} expect={expected} got={got} reason={reason}")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
