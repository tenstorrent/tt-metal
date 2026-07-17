# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared performance-metric formulas for Tensix hardware counters.

Single source of truth for the derived metrics, consumed by BOTH:
  - tt-llk test harness  (helpers/metrics.py, from the LLK counter dump)
  - tt-metal Tracy tool  (tools/tracy/perf_counter_analysis.py, from a Tracy capture)

Each path adapts its counter data to the small CounterView interface below, then calls
compute_metrics(view).

Two metric families, both expressed as `_pct` (×100):
  - Bounded utilizations/efficiencies/rates (0-100%): single-port fractions are busy/ref cycles;
    multi-port client groups (L1/NoC) report the MEAN per-port utilization over their PRESENT ports
    (see mean_port_util) rather than a summed rate, so none scales with port count or exceeds 100%.
  - Unbounded ratios (may exceed 100% by design): math_to_pack_handoff (available-math per busy
    packer — >100% means the packer is the bottleneck); stall_overlap_t* (summed per-resource wait
    rate — >100% means several waits overlap in the same cycle); unpacker_l1_eff / packer_l1_eff
    (L1-bank grant cycles per compute-bank busy cycle — cross-domain, so >100% means the L1 port is
    granted more than the engine reports busy, e.g. WH's multi-packer); and compute_to_unpack_ratio
    (math/SFPU ops per unpacker-busy cycle — >100% = compute-bound, e.g. unary SFPU can reach several
    hundred %). These are genuine ratios, not clamped, and are commented as such at their definitions.

Arch isolation: the counter inventory is per-arch (hw_counters.h), so on a given machine only that
arch's counters exist. compute_metrics is the arch UNION and gates every arch-divergent metric on
CounterView.has(): a metric whose counters are absent on the current arch returns None (NOT 0.0),
so WH-only metrics (per-engine packers) read N/A on Blackhole and BH-only metrics (extended L1
ports) read N/A on Wormhole.
"""

import re
from pathlib import Path
from typing import Protocol


def perf_counter_type_names(hpp_path=None) -> dict:
    """Parse the ``PerfCounterType`` enum from perf_counters.hpp into ``{ordinal: name}``.

    Single source for the Tracy decode table (the profiler stores the compiled enum ordinal in the
    counter_type field), so it can't drift from the C++ enum. Defaults to the sibling perf_counters.hpp.
    """
    if hpp_path is None:
        hpp_path = Path(__file__).resolve().parent / "perf_counters.hpp"
    body = re.search(r"enum\s+PerfCounterType[^{]*\{(.*?)\};", Path(hpp_path).read_text(), re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)  # drop line comments
    names = {}
    val = -1
    for tok in body.split(","):
        tok = tok.strip()
        m = re.match(r"([A-Za-z_]\w*)\s*(?:=\s*(\d+))?$", tok) if tok else None
        if not m:
            continue
        val = int(m.group(2)) if m.group(2) else val + 1
        names[val] = m.group(1)
    return names


class CounterView(Protocol):
    """Minimal counter accessor each data path adapts to (bank/name identity + per-bank cycles)."""

    def count(self, bank: str, counter_name: str) -> float:
        """Average count for a counter (0.0 if absent)."""
        ...

    def cycles(self, bank: str) -> float:
        """Average reference-cycle count for a bank (0.0 if absent)."""
        ...

    def has(self, counter_name: str) -> bool:
        """Whether a counter is present in the data."""
        ...


def safe_div(numerator: float, denominator: float) -> "float | None":
    """Safe division returning None if denominator is 0."""
    return (numerator / denominator) if denominator > 0 else None


def pct(value: "float | None") -> "float | None":
    """Convert ratio to percentage."""
    return (value * 100.0) if value is not None else None


def one_minus(value: "float | None") -> "float | None":
    """Compute 1.0 - value, for inverting 'not stalled' into 'stalled'."""
    return (1.0 - value) if value is not None else None


def avg_pair(a: "float | None", b: "float | None") -> "float | None":
    """Average of two optional values."""
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def mean_port_util(v: "CounterView", bank: str, names, cycles: float) -> "float | None":
    """
    Mean per-port utilization across a client group, bounded 0-1.

    Each port's utilization is (busy cycles / ref cycles) ∈ [0, 1]; the group metric is the mean over
    only the PRESENT ports (ports absent on an arch are excluded from the count, not treated as 0), so
    it stays a true 0-100% utilization instead of an unbounded sum. Returns None if no port is present.
    """
    present = [n for n in names if v.has(n)]
    if not present:
        return None
    return safe_div(sum(v.count(bank, n) for n in present), len(present) * cycles)


# ── L1 client-port groupings (single source; both the Tracy tool and the LLK harness reference these ──
# instead of hardcoding port lists, which is where the names drift). Each group is the arch UNION — callers
# filter to the ports actually present in their data (see mean_port_util / has()).
L1_RING0 = (
    "L1_0_NOC_RING0_OUTGOING_0",
    "L1_0_NOC_RING0_OUTGOING_1",
    "L1_0_NOC_RING0_INCOMING_0",
    "L1_0_NOC_RING0_INCOMING_1",
    "L1_2_NOC_RING0_OUTGOING_2",
    "L1_2_NOC_RING0_OUTGOING_3",
    "L1_2_NOC_RING0_INCOMING_2",
    "L1_2_NOC_RING0_INCOMING_3",
)
L1_RING1 = (
    "L1_1_NOC_RING1_OUTGOING_0",
    "L1_1_NOC_RING1_OUTGOING_1",
    "L1_1_NOC_RING1_INCOMING_0",
    "L1_1_NOC_RING1_INCOMING_1",
    "L1_3_NOC_RING1_OUTGOING_2",
    "L1_3_NOC_RING1_OUTGOING_3",
    "L1_3_NOC_RING1_INCOMING_2",
    "L1_3_NOC_RING1_INCOMING_3",
)
# L1_0 port 1 (the packer/ECC client) is named differently per arch: WH _ECC_PACK1, BH _ECC.
L1_UNPACKER_PORT1_NAMES = ("L1_0_UNPACKER_1_ECC_PACK1", "L1_0_UNPACKER_1_ECC")
L1_UNPACKER = ("L1_0_UNPACKER_0",) + L1_UNPACKER_PORT1_NAMES
# Extended unpacker interfaces: L1_1 (both arches) + L1_2 (BH only, ext 4-7).
L1_EXT_UNPACKER = tuple(f"L1_1_EXT_UNPACKER_{i}" for i in (1, 2, 3)) + tuple(
    f"L1_2_EXT_UNPACKER_{i}" for i in (4, 5, 6, 7)
)
# Extended TDMA packer interfaces (BH): L1_3 (0-3) + L1_4 (4-5).
L1_EXT_PACK = tuple(f"L1_3_TDMA_PACK_EXT_{i}" for i in (0, 1, 2, 3)) + tuple(f"L1_4_TDMA_PACK_EXT_{i}" for i in (4, 5))
L1_TAG_SEARCH = ("L1_4_TAG_SEARCH_PACKER1",)
L1_TDMA_BUNDLE = ("L1_0_TDMA_BUNDLE_0_RISC", "L1_0_TDMA_BUNDLE_1_TRISC")
L1_ALL = L1_RING0 + L1_RING1 + L1_UNPACKER + L1_EXT_UNPACKER + L1_EXT_PACK + L1_TAG_SEARCH + L1_TDMA_BUNDLE


def l1_unpacker_port1_name(has) -> "str | None":
    """Present arch-specific L1_0 port-1 counter name (WH `_ECC_PACK1` / BH `_ECC`), or None.

    `has` is a predicate `name -> bool`. Callers use this instead of hardcoding the WH name (which
    silently misses the BH port). See L1_UNPACKER_PORT1_NAMES.
    """
    for name in L1_UNPACKER_PORT1_NAMES:
        if has(name):
            return name
    return None


def compute_metrics(v: CounterView) -> dict:
    """Compute all derived metrics from a counter view. Returns a flat dict of *_pct values."""
    # ── Reference cycles per bank ──
    fpu_cycles = v.cycles("FPU")
    instrn_cycles = v.cycles("INSTRN_THREAD")
    pack_cycles = v.cycles("TDMA_PACK")
    l1_cycles = v.cycles("L1")

    # Compute Utilization (FPU bank). Counter names mirror tt-metal PerfCounterType.
    fpu_instruction = v.count("FPU", "FPU_COUNTER")
    fpu_or_sfpu = v.count("FPU", "MATH_COUNTER")
    fpu_utilization = safe_div(fpu_instruction, fpu_cycles)
    compute_utilization = safe_div(fpu_or_sfpu, fpu_cycles)

    # ── Thread Stall Rates (INSTRN_THREAD bank) ──
    stalls_0 = v.count("INSTRN_THREAD", "THREAD_STALLS_0")
    stalls_1 = v.count("INSTRN_THREAD", "THREAD_STALLS_1")
    stalls_2 = v.count("INSTRN_THREAD", "THREAD_STALLS_2")
    unpack_thread_stall = safe_div(stalls_0, instrn_cycles)
    math_thread_stall = safe_div(stalls_1, instrn_cycles)
    pack_thread_stall = safe_div(stalls_2, instrn_cycles)

    # ── Semaphore Wait Rates (INSTRN_THREAD bank) ──
    sem_wait_1 = v.count("INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_1")
    sem_wait_2 = v.count("INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_2")
    math_sem_wait = safe_div(sem_wait_1, instrn_cycles)
    pack_sem_wait = safe_div(sem_wait_2, instrn_cycles)

    # ── Unpacker Write Efficiency (TDMA_UNPACK bank) ──
    srca_write = v.count("TDMA_UNPACK", "SRCA_WRITE_ACTUAL")
    srcb_write = v.count("TDMA_UNPACK", "SRCB_WRITE_ACTUAL")
    unpack0_busy = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    unpack0_eff = safe_div(srca_write, unpack0_busy)
    unpack1_eff = safe_div(srcb_write, unpack1_busy)
    unpack_eff = avg_pair(unpack0_eff, unpack1_eff)

    # ── Unpacker-to-Math Data Flow (TDMA_UNPACK bank) ──
    srca_avail = v.count("TDMA_UNPACK", "SRCA_WRITE_AVAILABLE")
    srcb_avail = v.count("TDMA_UNPACK", "SRCB_WRITE_AVAILABLE")
    flow0 = safe_div(srca_avail, unpack0_busy)
    flow1 = safe_div(srcb_avail, unpack1_busy)
    flow_avg = avg_pair(flow0, flow1)

    # Packer Metrics — aggregate IDs work on both WH (per-engine also exposed) and BH (single packer).
    packer_busy = v.count("TDMA_PACK", "PACKER_BUSY")
    pack_utilization = safe_div(packer_busy, pack_cycles)
    dest_read = v.count("TDMA_PACK", "PACKER_DEST_READ_AVAILABLE")
    pack_dest_eff = safe_div(dest_read, packer_busy)

    # ── Math Pipeline Stalls (TDMA_UNPACK bank only — same bank, reliable) ──
    math_available = v.count("TDMA_UNPACK", "MATH_INSTRN_AVAILABLE")
    math_not_blocked = v.count("TDMA_UNPACK", "MATH_SRC_DATA_READY")

    # Math src data stall: fraction of math-available cycles where src data was NOT ready
    math_src_stall_rate = one_minus(safe_div(math_not_blocked, math_available))

    # ── L1 / NoC utilization: mean per-port busy fraction across each client group (bounded 0-100%). ──
    # A client group's ports each run at 0-100% (busy/ref cycles); the group metric is the mean over its
    # PRESENT ports (ports absent on an arch are excluded from the divisor, not counted as 0). This keeps
    # every value a true utilization instead of a summed rate that scales with port count and exceeds 100%.
    # Client-port groupings live at module level (L1_RING0 etc.) so the Tracy tool shares the same lists.
    noc_ring0_util = mean_port_util(v, "L1", L1_RING0, l1_cycles)
    noc_ring1_util = mean_port_util(v, "L1", L1_RING1, l1_cycles)
    unpacker_l1_util = mean_port_util(v, "L1", L1_UNPACKER, l1_cycles)
    ext_unpacker_l1_util = mean_port_util(v, "L1", L1_EXT_UNPACKER, l1_cycles)
    ext_pack_l1_util = mean_port_util(v, "L1", L1_EXT_PACK, l1_cycles)
    tag_search_l1_util = mean_port_util(v, "L1", L1_TAG_SEARCH, l1_cycles)
    tdma_bundle_l1_util = mean_port_util(v, "L1", L1_TDMA_BUNDLE, l1_cycles)
    l1_mean_client_util = mean_port_util(v, "L1", L1_ALL, l1_cycles)
    # NoC ring0 grant efficiency: fraction of requests served (grant <= req, already bounded 0-100%).
    _ring0_req = sum(v.count("L1", c) for c in L1_RING0 if v.has(c))
    _ring0_grant = sum(v.count("L1", c + "_GRANT") for c in L1_RING0 if v.has(c + "_GRANT"))
    noc_ring0_grant_eff = safe_div(_ring0_grant, _ring0_req)

    # ── Per-thread instruction throughput (INSTRN bank) ──
    thread0_ipc = safe_div(v.count("INSTRN_THREAD", "THREAD_INSTRUCTIONS_0"), instrn_cycles)
    thread1_ipc = safe_div(v.count("INSTRN_THREAD", "THREAD_INSTRUCTIONS_1"), instrn_cycles)
    thread2_ipc = safe_div(v.count("INSTRN_THREAD", "THREAD_INSTRUCTIONS_2"), instrn_cycles)

    # ── Cross-thread dependency stalls (INSTRN per-thread WAITING_FOR_*) ──
    # Where each pipeline stage blocks: math starved by unpack, pack blocked on math, etc.
    math_wait_unpack = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_UNPACK_IDLE_1"), instrn_cycles)
    math_wait_sfpu = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_SFPU_IDLE_1"), instrn_cycles)
    pack_wait_math = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_MATH_IDLE_2"), instrn_cycles)
    unpack_wait_pack = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_PACK_IDLE_0"), instrn_cycles)
    math_wait_srca = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_SRCA_VALID"), instrn_cycles)
    math_wait_srcb = safe_div(v.count("INSTRN_THREAD", "WAITING_FOR_SRCB_VALID"), instrn_cycles)

    # ── Per-engine packer (TDMA_PACK; WH exposes 4 engines, BH a single packer → others N/A) ──
    pb = [
        v.count("TDMA_PACK", "PACKER_BUSY_0"),
        v.count("TDMA_PACK", "PACKER_BUSY_1"),
        v.count("TDMA_PACK", "PACKER_BUSY_2"),
        v.count("TDMA_PACK", "PACKER_BUSY"),  # engine 3 (see hw_counters naming note)
    ]
    # Per-engine packers 0-2 are WH-only (PACKER_BUSY_0/1/2). Gate on has() so BH reads N/A (None)
    # instead of a misleading 0% — count() returns 0.0 for an absent counter, which safe_div would
    # otherwise turn into a real 0%. Engine 3 uses PACKER_BUSY (present on both arches).
    packer0_util = safe_div(pb[0], pack_cycles) if v.has("PACKER_BUSY_0") else None
    packer1_util = safe_div(pb[1], pack_cycles) if v.has("PACKER_BUSY_1") else None
    packer2_util = safe_div(pb[2], pack_cycles) if v.has("PACKER_BUSY_2") else None
    packer3_util = safe_div(pb[3], pack_cycles)
    active_pb = [b for b in pb if b > 0]
    packer_imbalance = safe_div(max(active_pb) - min(active_pb), max(active_pb)) if len(active_pb) >= 2 else None
    dest_granted = v.count("TDMA_PACK", "DEST_READ_GRANTED_0")
    pack_dest_grant_eff = safe_div(dest_granted, dest_read)

    # ── Source-register write completion efficiency (TDMA_UNPACK) ──
    srca_write_eff = safe_div(srca_write, srca_avail)
    srcb_write_eff = safe_div(srcb_write, srcb_avail)

    # ── Math pipeline stall breakdown (fraction of math-available cycles stalled by each cause) ──
    # The counters count "not stalled" cycles, so each stall rate is the complement.
    # NOTE: math_dest_wr_port_stall / math_scoreboard_stall take their numerator from the TDMA_PACK
    # bank but normalize by math_available (TDMA_UNPACK). Gate on has(): a capture with the unpack
    # group but NOT the pack group would otherwise give count()=0 -> safe_div(0, +)=0 -> one_minus=1.0,
    # i.e. a bogus 100% stall instead of N/A (None). (data_hazard_stall is same-bank, no gate needed.)
    data_hazard_stall = one_minus(safe_div(v.count("TDMA_UNPACK", "DATA_HAZARD_STALLS_MOVD2A"), math_available))
    math_dest_wr_port_stall = (
        one_minus(safe_div(v.count("TDMA_PACK", "MATH_NOT_STALLED_DEST_WR_PORT"), math_available))
        if v.has("MATH_NOT_STALLED_DEST_WR_PORT")
        else None
    )
    math_scoreboard_stall = (
        one_minus(safe_div(v.count("TDMA_PACK", "AVAILABLE_MATH"), math_available)) if v.has("AVAILABLE_MATH") else None
    )
    math_pipeline_util = safe_div(v.count("TDMA_UNPACK", "MATH_INSTRN_STARTED"), math_available)

    # ══ Superset metrics (also consumed by the Tracy tool) ═════════════════════════════════════════
    # NOTE: Tracy historically applied a few of these per-core with cross-core guards (e.g. "skip if
    # sum()==0", "skip unless median grant/req > 0.1"). Here they are pure per-view functions; safe_div
    # returns None when a denominator is 0, which covers the degenerate cases without cross-view state.
    port1 = l1_unpacker_port1_name(v.has)  # arch L1_0 port-1: WH _ECC_PACK1 / BH _ECC

    # ── Compute (extra) ──
    sfpu_util = safe_div(v.count("FPU", "SFPU_COUNTER"), fpu_cycles)
    fpu_exec_eff = safe_div(fpu_instruction, v.count("INSTRN_THREAD", "FPU_INSTRN_AVAILABLE_1"))
    # Math→pack handoff (UNBOUNDED ratio): available-math per busy packer; fall back to bank cycles
    # when packer idle. >100% is legitimate and meaningful — it means more math-result cycles were
    # available than the packer consumed, i.e. the packer is the handoff bottleneck. Not clamped.
    available_math = v.count("TDMA_PACK", "AVAILABLE_MATH")
    math_to_pack_handoff = (
        safe_div(available_math, packer_busy) if packer_busy > 0 else safe_div(available_math, pack_cycles)
    )

    # ── Extra INSTRN waits / availability (all / instrn_cycles) ──
    def _instrn_rate(name):
        return safe_div(v.count("INSTRN_THREAD", name), instrn_cycles)

    srca_clear_wait = _instrn_rate("WAITING_FOR_SRCA_CLEAR")
    srcb_clear_wait = _instrn_rate("WAITING_FOR_SRCB_CLEAR")
    math_idle_wait_t1 = _instrn_rate("WAITING_FOR_MATH_IDLE_1")
    pack_idle_wait_t2 = _instrn_rate("WAITING_FOR_PACK_IDLE_2")
    unpack_idle_wait_t0 = _instrn_rate("WAITING_FOR_UNPACK_IDLE_0")
    sem_zero_wait_t0 = _instrn_rate("WAITING_FOR_NONZERO_SEM_0")
    sem_full_wait_t0 = _instrn_rate("WAITING_FOR_NONFULL_SEM_0")
    sem_full_wait_t1 = _instrn_rate("WAITING_FOR_NONFULL_SEM_1")
    sem_full_wait_t2 = _instrn_rate("WAITING_FOR_NONFULL_SEM_2")
    mmio_idle_wait_t0 = _instrn_rate("WAITING_FOR_MMIO_IDLE_0")
    thcon_idle_wait_t0 = _instrn_rate("WAITING_FOR_THCON_IDLE_0")
    move_idle_wait_t0 = _instrn_rate("WAITING_FOR_MOVE_IDLE_0")
    cfg_instrn_avail_t0 = _instrn_rate("CFG_INSTRN_AVAILABLE_0")
    sync_instrn_avail_t0 = _instrn_rate("SYNC_INSTRN_AVAILABLE_0")
    thcon_instrn_avail_t0 = _instrn_rate("THCON_INSTRN_AVAILABLE_0")
    move_instrn_avail_t0 = _instrn_rate("MOVE_INSTRN_AVAILABLE_0")
    math_instrn_avail_t1 = _instrn_rate("FPU_INSTRN_AVAILABLE_1")
    unpack_instrn_avail_t0 = _instrn_rate("UNPACK_INSTRN_AVAILABLE_0")
    pack_instrn_avail_t2 = _instrn_rate("PACK_INSTRN_AVAILABLE_2")

    # ── Write-blocked complements (TDMA_UNPACK: fraction of available writes NOT completed) ──
    srca_write_port_blocked = one_minus(safe_div(srca_write, srca_avail))
    srca_write_ovr_blocked = one_minus(safe_div(v.count("TDMA_UNPACK", "SRCA_WRITE_NOT_BLOCKED_OVR"), srca_avail))
    srcb_write_ovr_blocked = one_minus(safe_div(srcb_write, srcb_avail))
    srcb_write_port_blocked = one_minus(safe_div(v.count("TDMA_UNPACK", "SRCB_WRITE_NOT_BLOCKED_PORT"), srcb_avail))
    dest_read_backpressure = one_minus(safe_div(dest_granted, dest_read))

    # ── Instruction issue rates are the IPC metrics above (thread{0,1,2}_ipc). ──

    # ── L1 per-port utilisation + grant efficiency (extra) ──
    risc_core_l1_util = safe_div(v.count("L1", "L1_0_TDMA_BUNDLE_0_RISC"), l1_cycles)
    l1_packer_port_util = safe_div(v.count("L1", port1), l1_cycles) if port1 else None
    # UNBOUNDED ratios: L1-bank grant cycles per compute-bank busy cycle. The grant counter (L1 bank)
    # and the busy counter (TDMA_UNPACK/TDMA_PACK bank) are from different measurement domains, so the
    # ratio can exceed 100% when the L1 port is granted more cycles than the engine reports busy (e.g.
    # WH's multi-packer, where L1_0_PORT1_GRANT > aggregate PACKER_BUSY). >100% = ample L1 bandwidth.
    unpacker_l1_eff = safe_div(v.count("L1", "L1_0_UNPACKER_0_GRANT"), unpack0_busy)
    packer_l1_eff = safe_div(v.count("L1", "L1_0_PORT1_GRANT"), packer_busy)
    l1_unpacker_backpressure = one_minus(
        safe_div(v.count("L1", "L1_0_UNPACKER_0_GRANT"), v.count("L1", "L1_0_UNPACKER_0"))
    )
    l1_packer_port_backpressure = (
        one_minus(safe_div(v.count("L1", "L1_0_PORT1_GRANT"), v.count("L1", port1))) if port1 else None
    )

    # ── NoC ring back-pressure (mean (req-grant)/req over the ring's outgoing/incoming ports) ──
    def _bp(names):
        req = sum(v.count("L1", n) for n in names if v.has(n))
        grant = sum(v.count("L1", n + "_GRANT") for n in names if v.has(n + "_GRANT"))
        return one_minus(safe_div(grant, req))

    _R0_OUT = ("L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1")
    _R0_IN = ("L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1")
    _R1_OUT = ("L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1")
    _R1_IN = ("L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1")
    noc_ring0_out_bp = _bp(_R0_OUT)
    noc_ring0_in_bp = _bp(_R0_IN)
    noc_ring1_out_bp = _bp(_R1_OUT)
    noc_ring1_in_bp = _bp(_R1_IN)

    # Split NoC utilisation (per direction, primary L1_0/L1_1 channels only) — Tracy reports these
    # separately; the merged noc_ring{0,1}_util above additionally include the BH secondary channels.
    noc_ring0_out_util = mean_port_util(v, "L1", _R0_OUT, l1_cycles)
    noc_ring0_in_util = mean_port_util(v, "L1", _R0_IN, l1_cycles)
    noc_ring1_out_util = mean_port_util(v, "L1", _R1_OUT, l1_cycles)
    noc_ring1_in_util = mean_port_util(v, "L1", _R1_IN, l1_cycles)

    # ── L1 bank-0 composite balance metrics (over the 8 primary L1_0 ports) ──
    _unp0 = v.count("L1", "L1_0_UNPACKER_0")
    _pk = v.count("L1", port1) if port1 else 0.0
    _bundle = v.count("L1", "L1_0_TDMA_BUNDLE_0_RISC") + v.count("L1", "L1_0_TDMA_BUNDLE_1_TRISC")
    _noc_out = sum(v.count("L1", n) for n in _R0_OUT)
    _noc_in = sum(v.count("L1", n) for n in _R0_IN)
    _l1_total = _unp0 + _pk + _bundle + _noc_out + _noc_in
    l1_total_bw = safe_div(_l1_total, 8 * l1_cycles)
    _reads = _unp0 + _noc_out
    _writes = _pk + _noc_in
    l1_read_write_ratio = safe_div(_reads, _reads + _writes)
    noc_ring0_asymmetry = safe_div(_noc_out, _noc_out + _noc_in)
    tdma_vs_noc_l1_share = safe_div(_bundle, _bundle + _noc_out + _noc_in)
    # Contention index: mean back-pressure over the 5 primary request/grant port pairs.
    _CONTENTION = ("L1_0_UNPACKER_0",) + _R0_OUT + _R0_IN
    _c_bps = [
        one_minus(safe_div(v.count("L1", n + "_GRANT"), v.count("L1", n)))
        for n in _CONTENTION
        if v.has(n) and v.has(n + "_GRANT")
    ]
    _c_bps = [b for b in _c_bps if b is not None]
    l1_contention_index = (sum(_c_bps) / len(_c_bps)) if _c_bps else None
    # NoC-vs-compute balance: NoC ring0 traffic vs FPU work.
    _noc_total = _noc_out + _noc_in
    noc_vs_compute_balance = safe_div(_noc_total, fpu_instruction + _noc_total)

    # ── Stall-cause overlap: summed per-resource wait rate for thread t (UNBOUNDED ratio) ──
    # Normalized by instrn_cycles (fraction of thread cycles spent in ANY wait), NOT by
    # THREAD_STALLS_t. THREAD_STALLS_t counts a narrower "thread fully stalled" event that is
    # incommensurate with the per-resource WAITING_FOR_* cycles — dividing by it produced absurd
    # values (e.g. 1210 wait-cycles / 48 stall-cycles = 2521%). With instrn_cycles the value is a
    # meaningful rate that only exceeds 100% when several waits overlap in the same cycle (that
    # overlap is the signal, so it is intentionally not clamped).
    def _stall_overlap(t):
        reasons = [
            f"WAITING_FOR_THCON_IDLE_{t}",
            f"WAITING_FOR_UNPACK_IDLE_{t}",
            f"WAITING_FOR_PACK_IDLE_{t}",
            f"WAITING_FOR_MATH_IDLE_{t}",
            f"WAITING_FOR_NONZERO_SEM_{t}",
            f"WAITING_FOR_NONFULL_SEM_{t}",
            f"WAITING_FOR_MOVE_IDLE_{t}",
            f"WAITING_FOR_MMIO_IDLE_{t}",
            f"WAITING_FOR_SFPU_IDLE_{t}",
        ]
        return safe_div(sum(v.count("INSTRN_THREAD", r) for r in reasons), instrn_cycles)

    stall_overlap_t0 = _stall_overlap(0)
    stall_overlap_t1 = _stall_overlap(1)
    stall_overlap_t2 = _stall_overlap(2)

    # ── Compute-to-unpack ratio (UNBOUNDED) ── math/SFPU ops per unpacker-busy cycle. >100% =
    # compute-bound (e.g. unary SFPU: heavy compute, minimal unpack → can reach several hundred %);
    # <100% = memory/unpack-bound. A genuine ratio, not clamped.
    compute_to_unpack = safe_div(fpu_or_sfpu, unpack0_busy + unpack1_busy)

    # ══ Additional derivable metrics ═══════════════════════════════════════════════════════════════
    # NoC ring-1 grant efficiency (mirror of ring-0; all counters present on both arches).
    _ring1_req = sum(v.count("L1", c) for c in L1_RING1 if v.has(c))
    _ring1_grant = sum(v.count("L1", c + "_GRANT") for c in L1_RING1 if v.has(c + "_GRANT"))
    noc_ring1_grant_eff = safe_div(_ring1_grant, _ring1_req)

    # Aggregate cross-thread stall indicator (ANY_THREAD_STALL / instrn cycles), distinct from the
    # summed per-thread THREAD_STALLS_* — a single "was any thread stalled this cycle" rate.
    any_thread_stall = _instrn_rate("ANY_THREAD_STALL")

    # Extended L1 client back-pressure ((req-grant)/req). BH-only ports → None on WH (has()-gated
    # inside _bp via the present-name filter), matching the existing ext-* utilization metrics.
    l1_ext_unpacker_backpressure = _bp(L1_EXT_UNPACKER)
    l1_ext_pack_backpressure = _bp(L1_EXT_PACK)
    l1_tag_search_backpressure = _bp(L1_TAG_SEARCH)

    # Per-thread unpacker / src-write distribution (fraction driven by the math thread T1; the
    # per-thread counters exist on both arches). Bounded 0-1 (x/(x+y)); None if that unpacker idle.
    _u0_t0 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    _u0_t1 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD1")
    _u1_t0 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    _u1_t1 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD1")
    unpack0_thread1_share = safe_div(_u0_t1, _u0_t0 + _u0_t1)
    unpack1_thread1_share = safe_div(_u1_t1, _u1_t0 + _u1_t1)
    _sa_t0 = v.count("TDMA_UNPACK", "SRCA_WRITE_THREAD0")
    _sa_t1 = v.count("TDMA_UNPACK", "SRCA_WRITE_THREAD1")
    _sb_t0 = v.count("TDMA_UNPACK", "SRCB_WRITE_THREAD0")
    _sb_t1 = v.count("TDMA_UNPACK", "SRCB_WRITE_THREAD1")
    srca_write_thread0_share = safe_div(_sa_t0, _sa_t0 + _sa_t1)
    srcb_write_thread0_share = safe_div(_sb_t0, _sb_t0 + _sb_t1)

    return {
        # Compute utilization
        "fpu_utilization_pct": pct(fpu_utilization),
        "compute_utilization_pct": pct(compute_utilization),
        # Thread stall rates
        "unpack_thread_stall_pct": pct(unpack_thread_stall),
        "math_thread_stall_pct": pct(math_thread_stall),
        "pack_thread_stall_pct": pct(pack_thread_stall),
        # Semaphore waits
        "math_sem_wait_pct": pct(math_sem_wait),
        "pack_sem_wait_pct": pct(pack_sem_wait),
        # Unpacker write efficiency
        "unpack0_write_eff_pct": pct(unpack0_eff),
        "unpack1_write_eff_pct": pct(unpack1_eff),
        "unpack_write_eff_pct": pct(unpack_eff),
        # Unpacker-to-math flow
        "unpack_to_math_flow0_pct": pct(flow0),
        "unpack_to_math_flow1_pct": pct(flow1),
        "unpack_to_math_flow_pct": pct(flow_avg),
        # Packer metrics
        "pack_utilization_pct": pct(pack_utilization),
        "pack_dest_eff_pct": pct(pack_dest_eff),
        # Math pipeline stalls
        "math_src_stall_pct": pct(math_src_stall_rate),
        "data_hazard_stall_pct": pct(data_hazard_stall),
        "math_dest_wr_port_stall_pct": pct(math_dest_wr_port_stall),
        "math_scoreboard_stall_pct": pct(math_scoreboard_stall),
        "math_pipeline_util_pct": pct(math_pipeline_util),
        # L1 composite (mean utilization across all present L1 client ports)
        "l1_mean_client_util_pct": pct(l1_mean_client_util),
        # L1 / NoC utilization (mean per-port busy fraction within each client group)
        "noc_ring0_util_pct": pct(noc_ring0_util),
        "noc_ring1_util_pct": pct(noc_ring1_util),
        "noc_ring0_grant_eff_pct": pct(noc_ring0_grant_eff),
        "l1_unpacker_util_pct": pct(unpacker_l1_util),
        "l1_ext_unpacker_util_pct": pct(ext_unpacker_l1_util),
        "l1_ext_pack_util_pct": pct(ext_pack_l1_util),
        "l1_tag_search_util_pct": pct(tag_search_l1_util),
        "l1_tdma_bundle_util_pct": pct(tdma_bundle_l1_util),
        # Per-thread instruction throughput
        "thread0_ipc_pct": pct(thread0_ipc),
        "thread1_ipc_pct": pct(thread1_ipc),
        "thread2_ipc_pct": pct(thread2_ipc),
        # Cross-thread dependency stalls
        "math_wait_unpack_pct": pct(math_wait_unpack),
        "math_wait_sfpu_pct": pct(math_wait_sfpu),
        "pack_wait_math_pct": pct(pack_wait_math),
        "unpack_wait_pack_pct": pct(unpack_wait_pack),
        "math_wait_srca_pct": pct(math_wait_srca),
        "math_wait_srcb_pct": pct(math_wait_srcb),
        # Per-engine packer
        "packer0_util_pct": pct(packer0_util),
        "packer1_util_pct": pct(packer1_util),
        "packer2_util_pct": pct(packer2_util),
        "packer3_util_pct": pct(packer3_util),
        "packer_load_imbalance_pct": pct(packer_imbalance),
        "pack_dest_grant_eff_pct": pct(pack_dest_grant_eff),
        # Source-register write completion efficiency
        "srca_write_eff_pct": pct(srca_write_eff),
        "srcb_write_eff_pct": pct(srcb_write_eff),
        # ── Superset (also used by the Tracy tool) ──
        # Compute
        "sfpu_utilization_pct": pct(sfpu_util),
        "fpu_exec_eff_pct": pct(fpu_exec_eff),
        "math_to_pack_handoff_ratio": math_to_pack_handoff,
        # Extra INSTRN waits
        "srca_clear_wait_pct": pct(srca_clear_wait),
        "srcb_clear_wait_pct": pct(srcb_clear_wait),
        "math_idle_wait_t1_pct": pct(math_idle_wait_t1),
        "pack_idle_wait_t2_pct": pct(pack_idle_wait_t2),
        "unpack_idle_wait_t0_pct": pct(unpack_idle_wait_t0),
        "sem_zero_wait_t0_pct": pct(sem_zero_wait_t0),
        "sem_full_wait_t0_pct": pct(sem_full_wait_t0),
        "sem_full_wait_t1_pct": pct(sem_full_wait_t1),
        "sem_full_wait_t2_pct": pct(sem_full_wait_t2),
        "mmio_idle_wait_t0_pct": pct(mmio_idle_wait_t0),
        "thcon_idle_wait_t0_pct": pct(thcon_idle_wait_t0),
        "move_idle_wait_t0_pct": pct(move_idle_wait_t0),
        # Per-type instruction availability
        "cfg_instrn_avail_t0_pct": pct(cfg_instrn_avail_t0),
        "sync_instrn_avail_t0_pct": pct(sync_instrn_avail_t0),
        "thcon_instrn_avail_t0_pct": pct(thcon_instrn_avail_t0),
        "move_instrn_avail_t0_pct": pct(move_instrn_avail_t0),
        "math_instrn_avail_t1_pct": pct(math_instrn_avail_t1),
        "unpack_instrn_avail_t0_pct": pct(unpack_instrn_avail_t0),
        "pack_instrn_avail_t2_pct": pct(pack_instrn_avail_t2),
        # Write-blocked complements
        "srca_write_port_blocked_pct": pct(srca_write_port_blocked),
        "srca_write_ovr_blocked_pct": pct(srca_write_ovr_blocked),
        "srcb_write_ovr_blocked_pct": pct(srcb_write_ovr_blocked),
        "srcb_write_port_blocked_pct": pct(srcb_write_port_blocked),
        "dest_read_backpressure_pct": pct(dest_read_backpressure),
        # L1 per-port + grant efficiency
        "risc_core_l1_util_pct": pct(risc_core_l1_util),
        "l1_packer_port_util_pct": pct(l1_packer_port_util),
        "unpacker_l1_eff_ratio": unpacker_l1_eff,
        "packer_l1_eff_ratio": packer_l1_eff,
        "l1_unpacker_backpressure_pct": pct(l1_unpacker_backpressure),
        "l1_packer_port_backpressure_pct": pct(l1_packer_port_backpressure),
        # NoC ring back-pressure
        "noc_ring0_out_backpressure_pct": pct(noc_ring0_out_bp),
        "noc_ring0_in_backpressure_pct": pct(noc_ring0_in_bp),
        "noc_ring1_out_backpressure_pct": pct(noc_ring1_out_bp),
        "noc_ring1_in_backpressure_pct": pct(noc_ring1_in_bp),
        "noc_ring0_out_util_pct": pct(noc_ring0_out_util),
        "noc_ring0_in_util_pct": pct(noc_ring0_in_util),
        "noc_ring1_out_util_pct": pct(noc_ring1_out_util),
        "noc_ring1_in_util_pct": pct(noc_ring1_in_util),
        # L1 composites
        "l1_total_bw_pct": pct(l1_total_bw),
        "l1_read_write_ratio_pct": pct(l1_read_write_ratio),
        "noc_ring0_asymmetry_pct": pct(noc_ring0_asymmetry),
        "tdma_vs_noc_l1_share_pct": pct(tdma_vs_noc_l1_share),
        "l1_contention_index_pct": pct(l1_contention_index),
        "noc_vs_compute_balance_pct": pct(noc_vs_compute_balance),
        # Stall overlap + compute-to-unpack
        "stall_overlap_t0_ratio": stall_overlap_t0,
        "stall_overlap_t1_ratio": stall_overlap_t1,
        "stall_overlap_t2_ratio": stall_overlap_t2,
        "compute_to_unpack_ratio": compute_to_unpack,
        # Additional derivable metrics
        "noc_ring1_grant_eff_pct": pct(noc_ring1_grant_eff),
        "any_thread_stall_pct": pct(any_thread_stall),
        "l1_ext_unpacker_backpressure_pct": pct(l1_ext_unpacker_backpressure),
        "l1_ext_pack_backpressure_pct": pct(l1_ext_pack_backpressure),
        "l1_tag_search_backpressure_pct": pct(l1_tag_search_backpressure),
        "unpack0_thread1_share_pct": pct(unpack0_thread1_share),
        "unpack1_thread1_share_pct": pct(unpack1_thread1_share),
        "srca_write_thread0_share_pct": pct(srca_write_thread0_share),
        "srcb_write_thread0_share_pct": pct(srcb_write_thread0_share),
    }


# Human display name per metric key (the Tracy tool's historical names where they existed, so its CSV
# columns / summary stay stable). Consumers that print or write CSVs map keys through this.
METRIC_LABELS = {
    "fpu_utilization_pct": "FPU Util",
    "compute_utilization_pct": "MATH Util",
    "sfpu_utilization_pct": "SFPU Util",
    "fpu_exec_eff_pct": "FPU Execution Efficiency",
    "pack_utilization_pct": "Packer Utilization",
    "unpack0_write_eff_pct": "Unpacker0 Write Efficiency",
    "unpack1_write_eff_pct": "Unpacker1 Write Efficiency",
    "unpack_write_eff_pct": "Unpacker Write Efficiency",
    "pack_dest_eff_pct": "Packer Efficiency",
    "pack_dest_grant_eff_pct": "Pack Dest Grant Efficiency",
    "math_pipeline_util_pct": "Math Pipeline Utilization",
    "math_to_pack_handoff_ratio": "Math-to-Pack Handoff Efficiency",
    "unpack_to_math_flow_pct": "Unpacker-to-Math Data Flow",
    "unpack_to_math_flow0_pct": "Unpacker-to-Math Data Flow (srcA)",
    "unpack_to_math_flow1_pct": "Unpacker-to-Math Data Flow (srcB)",
    "math_src_stall_pct": "Math Src Data Stall Rate",
    "unpack_thread_stall_pct": "Thread 0 Stall Rate",
    "math_thread_stall_pct": "Thread 1 Stall Rate",
    "pack_thread_stall_pct": "Thread 2 Stall Rate",
    "math_wait_srca_pct": "SrcA Valid Wait",
    "math_wait_srcb_pct": "SrcB Valid Wait",
    "srca_clear_wait_pct": "SrcA Clear Wait",
    "srcb_clear_wait_pct": "SrcB Clear Wait",
    "math_idle_wait_t1_pct": "Math Idle Wait T1",
    "pack_idle_wait_t2_pct": "Pack Idle Wait T2",
    "unpack_idle_wait_t0_pct": "Unpack Idle Wait T0",
    "math_wait_unpack_pct": "Math Waiting on Unpack (T1)",
    "pack_wait_math_pct": "Pack Waiting on Math (T2)",
    "unpack_wait_pack_pct": "Unpack Waiting on Pack (T0)",
    "math_wait_sfpu_pct": "SFPU Idle Wait T1",
    "mmio_idle_wait_t0_pct": "MMIO Idle Wait T0",
    "thcon_idle_wait_t0_pct": "THCON Idle Wait T0",
    "move_idle_wait_t0_pct": "MOVE Idle Wait T0",
    "math_sem_wait_pct": "Semaphore Zero Wait T1",
    "pack_sem_wait_pct": "Semaphore Zero Wait T2",
    "sem_zero_wait_t0_pct": "Semaphore Zero Wait T0",
    "sem_full_wait_t0_pct": "Semaphore Full Wait T0",
    "sem_full_wait_t1_pct": "Semaphore Full Wait T1",
    "sem_full_wait_t2_pct": "Semaphore Full Wait T2",
    "cfg_instrn_avail_t0_pct": "CFG Instrn Avail Rate T0",
    "sync_instrn_avail_t0_pct": "SYNC Instrn Avail Rate T0",
    "thcon_instrn_avail_t0_pct": "THCON Instrn Avail Rate T0",
    "move_instrn_avail_t0_pct": "MOVE Instrn Avail Rate T0",
    "math_instrn_avail_t1_pct": "MATH Instrn Avail Rate T1",
    "unpack_instrn_avail_t0_pct": "UNPACK Instrn Avail Rate T0",
    "pack_instrn_avail_t2_pct": "PACK Instrn Avail Rate T2",
    "data_hazard_stall_pct": "Data Hazard Stall Rate",
    "math_dest_wr_port_stall_pct": "Math Dest Write Port Stall Rate",
    "math_scoreboard_stall_pct": "Math Scoreboard Stall Rate",
    "srca_write_eff_pct": "SrcA Write Actual Efficiency",
    "srcb_write_eff_pct": "SrcB Write Actual Efficiency",
    "srca_write_port_blocked_pct": "SrcA Write Port Blocked Rate",
    "srca_write_ovr_blocked_pct": "SrcA Write Overwrite Blocked Rate",
    "srcb_write_ovr_blocked_pct": "SrcB Write Overwrite Blocked Rate",
    "srcb_write_port_blocked_pct": "SrcB Write Port Blocked Rate",
    "dest_read_backpressure_pct": "Dest Read Backpressure",
    "thread0_ipc_pct": "T0 Instrn Issue Rate",
    "thread1_ipc_pct": "T1 Instrn Issue Rate",
    "thread2_ipc_pct": "T2 Instrn Issue Rate",
    "packer0_util_pct": "Packer Engine 0 Util",
    "packer1_util_pct": "Packer Engine 1 Util",
    "packer2_util_pct": "Packer Engine 2 Util",
    "packer3_util_pct": "Packer Engine 3 Util",
    "packer_load_imbalance_pct": "Packer Load Imbalance",
    "l1_unpacker_util_pct": "L1 Unpacker Port Util",
    "l1_packer_port_util_pct": "L1 Packer Port Util",
    "l1_tdma_bundle_util_pct": "L1 TDMA Bundle Util",
    "l1_ext_unpacker_util_pct": "L1 Ext Unpacker Util",
    "l1_ext_pack_util_pct": "L1 Ext Packer Util",
    "l1_tag_search_util_pct": "L1 Tag Search Util",
    "l1_mean_client_util_pct": "L1 Mean Client Util",
    "risc_core_l1_util_pct": "RISC Core L1 Util",
    "noc_ring0_util_pct": "NOC Ring 0 Util",
    "noc_ring1_util_pct": "NOC Ring 1 Util",
    "noc_ring0_out_util_pct": "NOC Ring 0 Outgoing Util",
    "noc_ring0_in_util_pct": "NOC Ring 0 Incoming Util",
    "noc_ring1_out_util_pct": "NOC Ring 1 Outgoing Util",
    "noc_ring1_in_util_pct": "NOC Ring 1 Incoming Util",
    "noc_ring0_grant_eff_pct": "NOC Ring 0 Grant Efficiency",
    "unpacker_l1_eff_ratio": "Unpacker L1 Efficiency",
    "packer_l1_eff_ratio": "Packer L1 Efficiency",
    "l1_unpacker_backpressure_pct": "L1 Unpacker Backpressure",
    "l1_packer_port_backpressure_pct": "L1 Packer Port Backpressure",
    "noc_ring0_out_backpressure_pct": "NOC Ring 0 Outgoing Backpressure",
    "noc_ring0_in_backpressure_pct": "NOC Ring 0 Incoming Backpressure",
    "noc_ring1_out_backpressure_pct": "NOC Ring 1 Outgoing Backpressure",
    "noc_ring1_in_backpressure_pct": "NOC Ring 1 Incoming Backpressure",
    "l1_total_bw_pct": "L1 Total Bandwidth Util",
    "l1_read_write_ratio_pct": "L1 Read vs Write Ratio",
    "noc_ring0_asymmetry_pct": "NOC Ring 0 Asymmetry",
    "tdma_vs_noc_l1_share_pct": "TDMA vs NOC L1 Share",
    "l1_contention_index_pct": "L1 Contention Index",
    "noc_vs_compute_balance_pct": "NOC vs Compute Balance",
    "stall_overlap_t0_ratio": "Stall Overlap T0",
    "stall_overlap_t1_ratio": "Stall Overlap T1",
    "stall_overlap_t2_ratio": "Stall Overlap T2",
    "compute_to_unpack_ratio": "Compute-to-Unpack Ratio",
    "noc_ring1_grant_eff_pct": "NOC Ring 1 Grant Efficiency",
    "any_thread_stall_pct": "Any-Thread Stall Rate",
    "l1_ext_unpacker_backpressure_pct": "L1 Ext Unpacker Backpressure",
    "l1_ext_pack_backpressure_pct": "L1 Ext Packer Backpressure",
    "l1_tag_search_backpressure_pct": "L1 Tag Search Backpressure",
    "unpack0_thread1_share_pct": "Unpacker0 T1 Share",
    "unpack1_thread1_share_pct": "Unpacker1 T1 Share",
    "srca_write_thread0_share_pct": "SrcA Write T0 Share",
    "srcb_write_thread0_share_pct": "SrcB Write T0 Share",
}


# ── Metric presentation classification (single source; consumers derive display units from this) ──
# Two families, distinguished by KEY SUFFIX:
#   *_pct   → bounded PERCENTAGE (0-100%), value already ×100. Numerator is a subset of the
#             denominator within one measurement domain (busy/ref, done/attempted, grant/req,
#             1-grant/req, x/(x+y)). Displayed with a "%" unit. This includes the per-thread
#             instruction-issue rates (single-issue Tensix ⇒ ≤1 instr/cycle ⇒ ≤100%).
#   *_ratio → unbounded RATIO, value is the RAW fraction (NOT ×100), may exceed 1.0 by design.
#             Numerator and denominator measure different quantities (cross-bank) or a sum can
#             exceed the reference. Displayed with a "ratio" unit (e.g. 1.48×), never clamped.
# RATIO_LABELS is the set of display names in the ratio family, for consumers that key by label.
RATIO_KEYS = {k for k in METRIC_LABELS if k.endswith("_ratio")}
RATIO_LABELS = {METRIC_LABELS[k] for k in RATIO_KEYS}
