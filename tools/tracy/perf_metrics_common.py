# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every perf-counter metric, shared by the Tracy tool and the LLK harness. Consumers adapt their data to
CounterView; absent counters read None, never 0. Keys end in _pct (bounded) or _ratio (unbounded)."""

import re
from pathlib import Path
from typing import Protocol


def perf_counter_type_names(hpp_path=None) -> dict:
    """Ordinal -> name table parsed from the PerfCounterType enum, so it cannot drift from the firmware."""
    if hpp_path is None:
        hpp_path = Path(__file__).resolve().parents[2] / "tt_metal" / "tools" / "profiler" / "perf_counters.hpp"
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

    def is_blackhole(self) -> bool:
        """Blackhole's L1_0 port 1 is an unpacker; Wormhole's carries pack1 traffic."""
        ...


def safe_div(numerator: float, denominator: float) -> "float | None":
    """Safe division returning None if denominator is 0."""
    return (numerator / denominator) if denominator > 0 else None


def pct(value: "float | None") -> "float | None":
    return (value * 100.0) if value is not None else None


def bounded(value: "float | None") -> "float | None":
    """Clamp a fraction to [0, 1]. The tt-1xx L1 grant counters are the interface ready line, not qualified
    by a request, so grant/request can exceed 1 when the interface sits ready with nothing to do."""
    return None if value is None else min(1.0, max(0.0, value))


def one_minus(value: "float | None") -> "float | None":
    return (1.0 - value) if value is not None else None


def avg_pair(a: "float | None", b: "float | None") -> "float | None":
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def first_present(v: "CounterView", names) -> "str | None":
    """The first of several per-arch names for one port that the data contains."""
    return next((n for n in names if v.has(n)), None)


def mean_port_util(v: "CounterView", bank: str, names, cycles: float) -> "float | None":
    """Mean busy fraction over the ports of one client group that are present in the data."""
    present = [n for n in names if v.has(n)]
    if not present:
        return None
    return safe_div(sum(v.count(bank, n) for n in present), len(present) * cycles)


# L1 client-port groupings, arch UNIONs; callers filter to the ports present in their data.
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
# Ports whose client differs per arch carry one name per arch; first_present() picks the one in the data.
# Port 1: pack1 + ECC on Wormhole, unpacker 1 + ECC on Blackhole. Port 8: packer L1 interface.
L1_PORT1_NAMES = ("L1_0_UNPACKER_1_ECC_PACK1", "L1_0_UNPACKER_1_ECC")
L1_PORT8_NAMES = ("L1_1_TDMA_PACKER_2", "L1_1_PACKER_IF_0")
L1_PORT1_GRANT = {"L1_0_UNPACKER_1_ECC_PACK1": "L1_0_PORT1_GRANT", "L1_0_UNPACKER_1_ECC": "L1_0_UNPACKER_1_ECC_GRANT"}
# Port 0 also carries the packer L1-to-L1 read on Blackhole.
L1_UNPACKER = ("L1_0_UNPACKER_0",)
# Extended read interfaces: unpacker 1 on banks 1-2 (ports 9-11, 16-19; also used by the packer L1-to-L1
# read), unpacker 0 on banks 4-5 (ports 35-41). One capture holds one bank, so each is a mean over what is present.
L1_UNPACKER1_EXT = (
    tuple(f"L1_1_EXT_UNPACKER_{i}" for i in (1, 2, 3))
    + tuple(f"L1_1_UNPACKER1_EXT_IF_{i}" for i in (1, 2, 3))
    + tuple(f"L1_2_UNPACKER1_EXT_IF_{i}" for i in (4, 5, 6, 7))
)
L1_UNPACKER0_EXT = tuple(f"L1_4_UNPACKER0_EXT_IF_{j}" for j in (1, 2, 3, 4, 5)) + tuple(
    f"L1_5_UNPACKER0_EXT_IF_{j}" for j in (6, 7)
)
L1_EXT_PACK = tuple(f"L1_3_EXT_PACKER_{i}" for i in (2, 3, 4, 5)) + tuple(f"L1_4_EXT_PACKER_{i}" for i in (6, 7))
L1_TAG_SEARCH = ("L1_4_TAG_SEARCH_PACKER_1",)
L1_TDMA_BUNDLE = ("L1_0_TDMA_BUNDLE_0_RISC", "L1_0_TDMA_BUNDLE_1_TRISC")
L1_ALL = (
    L1_RING0
    + L1_RING1
    + L1_UNPACKER
    + L1_PORT1_NAMES
    + L1_PORT8_NAMES
    + L1_UNPACKER1_EXT
    + L1_UNPACKER0_EXT
    + L1_EXT_PACK
    + L1_TAG_SEARCH
    + L1_TDMA_BUNDLE
)


def compute_metrics(v: CounterView) -> dict:
    """Every metric for one counter view; None wherever an input counter is absent."""
    fpu_cycles = v.cycles("FPU")
    instrn_cycles = v.cycles("INSTRN_THREAD")

    def _instrn_rate(name):
        """Rate over the INSTRN bank cycles; None when the counter was not captured (tt-2xx has no WAITING_FOR_*)."""
        return safe_div(v.count("INSTRN_THREAD", name), instrn_cycles) if v.has(name) else None

    pack_cycles = v.cycles("TDMA_PACK")
    l1_cycles = v.cycles("L1")

    fpu_instruction = v.count("FPU", "FPU_COUNTER")
    fpu_or_sfpu = v.count("FPU", "MATH_COUNTER")
    fpu_utilization = safe_div(fpu_instruction, fpu_cycles)
    compute_utilization = safe_div(fpu_or_sfpu, fpu_cycles)

    stalls_0 = v.count("INSTRN_THREAD", "THREAD_STALLS_0")
    stalls_1 = v.count("INSTRN_THREAD", "THREAD_STALLS_1")
    stalls_2 = v.count("INSTRN_THREAD", "THREAD_STALLS_2")
    unpack_thread_stall = safe_div(stalls_0, instrn_cycles)
    math_thread_stall = safe_div(stalls_1, instrn_cycles)
    pack_thread_stall = safe_div(stalls_2, instrn_cycles)

    sem_wait_1 = v.count("INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_1")
    sem_wait_2 = v.count("INSTRN_THREAD", "WAITING_FOR_NONZERO_SEM_2")
    math_sem_wait = safe_div(sem_wait_1, instrn_cycles)
    pack_sem_wait = safe_div(sem_wait_2, instrn_cycles)

    srca_write = v.count("TDMA_UNPACK", "SRCA_WRITE_NOT_BLOCKED_PORT")
    srcb_write = v.count("TDMA_UNPACK", "SRCB_WRITE_NOT_BLOCKED_OVR")
    unpack0_busy = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    unpack0_eff = safe_div(srca_write, unpack0_busy)
    unpack1_eff = safe_div(srcb_write, unpack1_busy)
    unpack_eff = avg_pair(unpack0_eff, unpack1_eff)

    srca_avail = v.count("TDMA_UNPACK", "SRCA_WRITE_REQ")
    srcb_avail = v.count("TDMA_UNPACK", "SRCB_WRITE_REQ")
    flow0 = safe_div(srca_avail, unpack0_busy)
    flow1 = safe_div(srcb_avail, unpack1_busy)
    flow_avg = avg_pair(flow0, flow1)

    # Packer Metrics: aggregate IDs work on both WH (per-engine also exposed) and BH (single packer).
    packer_busy = v.count("TDMA_PACK", "PACKER_BUSY")
    pack_utilization = safe_div(packer_busy, pack_cycles)
    dest_read = v.count("TDMA_PACK", "PACKER0_DEST_READ_REQ")
    pack_dest_eff = safe_div(dest_read, packer_busy)

    math_available = v.count("TDMA_UNPACK", "MATH_INSTRN_AVAILABLE")
    # No src-data stall metric: MATH_SRC_DATA_READY is gated on dec_instr_alu while
    # MATH_INSTRN_AVAILABLE counts the whole math pipe, so their ratio is not a stall fraction.

    noc_ring0_util = mean_port_util(v, "L1", L1_RING0, l1_cycles)
    noc_ring1_util = mean_port_util(v, "L1", L1_RING1, l1_cycles)
    unpacker_l1_util = mean_port_util(v, "L1", L1_UNPACKER, l1_cycles)
    unpacker1_ext_l1_util = mean_port_util(v, "L1", L1_UNPACKER1_EXT, l1_cycles)
    unpacker0_ext_l1_util = mean_port_util(v, "L1", L1_UNPACKER0_EXT, l1_cycles)
    ext_pack_l1_util = mean_port_util(v, "L1", L1_EXT_PACK, l1_cycles)
    tag_search_l1_util = mean_port_util(v, "L1", L1_TAG_SEARCH, l1_cycles)
    tdma_bundle_l1_util = mean_port_util(v, "L1", L1_TDMA_BUNDLE, l1_cycles)
    l1_mean_client_util = mean_port_util(v, "L1", L1_ALL, l1_cycles)
    # NoC ring0 grant efficiency: ready cycles per request cycle. The L1 grant counter is the interface
    # ready line, unqualified by the request, so this is clamped to [0, 1] (see bounded()).
    _ring0_req = sum(v.count("L1", c) for c in L1_RING0 if v.has(c))
    _ring0_grant = sum(v.count("L1", c + "_GRANT") for c in L1_RING0 if v.has(c + "_GRANT"))
    noc_ring0_grant_eff = bounded(safe_div(_ring0_grant, _ring0_req))

    thread0_ipc = _instrn_rate("THREAD_INSTRUCTIONS_0")
    thread1_ipc = _instrn_rate("THREAD_INSTRUCTIONS_1")
    thread2_ipc = _instrn_rate("THREAD_INSTRUCTIONS_2")

    math_wait_unpack = _instrn_rate("WAITING_FOR_UNPACK_IDLE_1")
    math_wait_sfpu = _instrn_rate("WAITING_FOR_SFPU_IDLE_1")
    pack_wait_math = _instrn_rate("WAITING_FOR_MATH_IDLE_2")
    unpack_wait_pack = _instrn_rate("WAITING_FOR_PACK_IDLE_0")
    math_wait_srca = _instrn_rate("WAITING_FOR_SRCA_VALID")
    math_wait_srcb = _instrn_rate("WAITING_FOR_SRCB_VALID")

    # ── Per-engine packer (TDMA_PACK; WH exposes 4 engines, BH a single packer → others N/A) ──
    pb = [
        v.count("TDMA_PACK", "PACKER_BUSY_0"),
        v.count("TDMA_PACK", "PACKER_BUSY_1"),
        v.count("TDMA_PACK", "PACKER_BUSY_2"),
        v.count("TDMA_PACK", "PACKER_BUSY"),  # engine 3 (see hw_counters naming note)
    ]
    # Engines 0-2 are Wormhole-only signals, so gate on has(); PACKER_BUSY is the whole packer on both arches.
    packer0_util = safe_div(pb[0], pack_cycles) if v.has("PACKER_BUSY_0") else None
    packer1_util = safe_div(pb[1], pack_cycles) if v.has("PACKER_BUSY_1") else None
    packer2_util = safe_div(pb[2], pack_cycles) if v.has("PACKER_BUSY_2") else None
    packer3_util = safe_div(pb[3], pack_cycles) if v.has("PACKER_BUSY_0") else None
    # Idle engines count as zero (100% imbalance), so gate on presence of all four, not on activity.
    _engines = ("PACKER_BUSY_0", "PACKER_BUSY_1", "PACKER_BUSY_2", "PACKER_BUSY")
    packer_imbalance = safe_div(max(pb) - min(pb), max(pb)) if all(v.has(n) for n in _engines) else None
    dest_granted = v.count("TDMA_PACK", "DEST_READ_GRANTED_0")
    pack_dest_grant_eff = safe_div(dest_granted, dest_read)

    srca_write_eff = safe_div(srca_write, srca_avail)
    srcb_write_eff = safe_div(srcb_write, srcb_avail)

    # Stall rates are complements of "not stalled" counters. The scoreboard one is cross-bank (numerator in
    # TDMA_PACK, denominator in TDMA_UNPACK), so it is has()-gated and clamped at 0.
    data_hazard_stall = one_minus(safe_div(v.count("TDMA_UNPACK", "MATH_NOT_D2S_STALLED"), math_available))
    math_scoreboard_stall = (
        bounded(one_minus(safe_div(v.count("TDMA_PACK", "MATH_NOT_SCOREBOARD_STALLED"), math_available)))
        if v.has("MATH_NOT_SCOREBOARD_STALLED")
        else None
    )
    math_pipeline_util = safe_div(v.count("TDMA_UNPACK", "MATH_INSTRN_STARTED"), math_available)

    l1_port1 = first_present(v, L1_PORT1_NAMES)
    l1_port8 = first_present(v, L1_PORT8_NAMES)

    sfpu_util = safe_div(v.count("FPU", "SFPU_COUNTER"), fpu_cycles)
    fpu_exec_eff = (
        safe_div(fpu_instruction, v.count("INSTRN_THREAD", "MATH_INSTRN_AVAILABLE_1")) if v.has("FPU_COUNTER") else None
    )
    # UNBOUNDED ratio: available-math per busy packer (bank cycles when idle); >1 means the packer
    # is the handoff bottleneck.
    available_math = v.count("TDMA_PACK", "MATH_NOT_SCOREBOARD_STALLED")
    math_to_pack_handoff = (
        safe_div(available_math, packer_busy) if packer_busy > 0 else safe_div(available_math, pack_cycles)
    )

    srca_clear_wait = _instrn_rate("WAITING_FOR_SRCA_CLEAR")
    srcb_clear_wait = _instrn_rate("WAITING_FOR_SRCB_CLEAR")
    math_idle_wait_t1 = _instrn_rate("WAITING_FOR_MATH_IDLE_1")
    pack_idle_wait_t2 = _instrn_rate("WAITING_FOR_PACK_IDLE_2")
    unpack_idle_wait_t0 = _instrn_rate("WAITING_FOR_UNPACK_IDLE_0")
    sem_zero_wait_t0 = _instrn_rate("WAITING_FOR_NONZERO_SEM_0")
    sem_full_wait_t0 = _instrn_rate("WAITING_FOR_NONFULL_SEM_0")
    sem_full_wait_t1 = _instrn_rate("WAITING_FOR_NONFULL_SEM_1")
    sem_full_wait_t2 = _instrn_rate("WAITING_FOR_NONFULL_SEM_2")
    cfg_idle_wait_t0 = _instrn_rate("WAITING_FOR_CFG_IDLE_0")
    thcon_idle_wait_t0 = _instrn_rate("WAITING_FOR_THCON_IDLE_0")
    move_idle_wait_t0 = _instrn_rate("WAITING_FOR_MOVE_IDLE_0")
    cfg_instrn_avail_t0 = _instrn_rate("CFG_INSTRN_AVAILABLE_0")
    sync_instrn_avail_t0 = _instrn_rate("SYNC_INSTRN_AVAILABLE_0")
    thcon_instrn_avail_t0 = _instrn_rate("THCON_INSTRN_AVAILABLE_0")
    move_instrn_avail_t0 = _instrn_rate("MOVE_INSTRN_AVAILABLE_0")
    math_instrn_avail_t1 = _instrn_rate("MATH_INSTRN_AVAILABLE_1")
    unpack_instrn_avail_t0 = _instrn_rate("UNPACK_INSTRN_AVAILABLE_0")
    pack_instrn_avail_t2 = _instrn_rate("PACK_INSTRN_AVAILABLE_2")

    srca_write_port_blocked = one_minus(safe_div(srca_write, srca_avail))
    srca_write_ovr_blocked = one_minus(safe_div(v.count("TDMA_UNPACK", "SRCA_WRITE_NOT_BLOCKED_OVR"), srca_avail))
    srcb_write_ovr_blocked = one_minus(safe_div(srcb_write, srcb_avail))
    srcb_write_port_blocked = one_minus(safe_div(v.count("TDMA_UNPACK", "SRCB_WRITE_NOT_BLOCKED_PORT"), srcb_avail))
    dest_read_backpressure = one_minus(safe_div(dest_granted, dest_read))

    risc_core_l1_util = (
        safe_div(v.count("L1", "L1_0_TDMA_BUNDLE_0_RISC"), l1_cycles) if v.has("L1_0_TDMA_BUNDLE_0_RISC") else None
    )
    l1_port1_util = safe_div(v.count("L1", l1_port1), l1_cycles) if l1_port1 else None
    l1_packer_port8_util = safe_div(v.count("L1", l1_port8), l1_cycles) if l1_port8 else None
    # UNBOUNDED ratios: L1 grant cycles per compute-engine busy cycle, cross-domain so >1 possible
    # (ample L1 bandwidth). packer_l1_eff is only meaningful on WH, where port 1 carries pack1 traffic.
    unpacker_l1_eff = (
        safe_div(v.count("L1", "L1_0_UNPACKER_0_GRANT"), unpack0_busy)
        if v.has("L1_0_UNPACKER_0_GRANT") and v.has("UNPACK0_BUSY_THREAD0")
        else None
    )
    packer_l1_eff = (
        safe_div(v.count("L1", "L1_0_PORT1_GRANT"), packer_busy)
        if v.has("L1_0_PORT1_GRANT") and v.has("PACKER_BUSY") and not v.is_blackhole()
        else None
    )  # Wormhole only: Blackhole port 1 carries no packer
    # Back-pressure = 1 - ready/request, clamped: the L1 grant counter is the ready line, see bounded().
    l1_unpacker_backpressure = bounded(
        one_minus(safe_div(v.count("L1", "L1_0_UNPACKER_0_GRANT"), v.count("L1", "L1_0_UNPACKER_0")))
    )
    l1_port1_backpressure = (
        bounded(one_minus(safe_div(v.count("L1", L1_PORT1_GRANT[l1_port1]), v.count("L1", l1_port1))))
        if l1_port1 and v.has(L1_PORT1_GRANT[l1_port1])
        else None
    )

    def _bp(names):
        req = sum(v.count("L1", n) for n in names if v.has(n))
        grant = sum(v.count("L1", n + "_GRANT") for n in names if v.has(n + "_GRANT"))
        return bounded(one_minus(safe_div(grant, req)))

    _R0_OUT = ("L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1")
    _R0_IN = ("L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1")
    _R1_OUT = ("L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1")
    _R1_IN = ("L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1")
    noc_ring0_out_bp = _bp(_R0_OUT)
    noc_ring0_in_bp = _bp(_R0_IN)
    noc_ring1_out_bp = _bp(_R1_OUT)
    noc_ring1_in_bp = _bp(_R1_IN)

    # Split NoC utilisation (per direction, primary L1_0/L1_1 channels only). Tracy reports these
    # separately; the merged noc_ring{0,1}_util above additionally include the BH secondary channels.
    noc_ring0_out_util = mean_port_util(v, "L1", _R0_OUT, l1_cycles)
    noc_ring0_in_util = mean_port_util(v, "L1", _R0_IN, l1_cycles)
    noc_ring1_out_util = mean_port_util(v, "L1", _R1_OUT, l1_cycles)
    noc_ring1_in_util = mean_port_util(v, "L1", _R1_IN, l1_cycles)

    _unp0 = v.count("L1", "L1_0_UNPACKER_0")
    _pk = v.count("L1", l1_port1) if l1_port1 else 0.0
    _bundle = v.count("L1", "L1_0_TDMA_BUNDLE_0_RISC") + v.count("L1", "L1_0_TDMA_BUNDLE_1_TRISC")
    _noc_out = sum(v.count("L1", n) for n in _R0_OUT)
    _noc_in = sum(v.count("L1", n) for n in _R0_IN)
    _l1_0 = v.has("L1_0_UNPACKER_0")  # the L1_0 bank was captured; otherwise these all read 0
    _l1_total = _unp0 + _pk + _bundle + _noc_out + _noc_in
    l1_total_bw = safe_div(_l1_total, 8 * l1_cycles) if _l1_0 else None
    # Port 1 is unpacker 1 on Blackhole (a read) and pack1 on Wormhole (a write).
    _bh = v.is_blackhole()
    _reads = _unp0 + _noc_out + (_pk if _bh else 0.0)
    _writes = _noc_in + (0.0 if _bh else _pk)
    l1_read_write_ratio = safe_div(_reads, _reads + _writes) if _l1_0 else None
    noc_ring0_asymmetry = safe_div(_noc_out, _noc_out + _noc_in) if _l1_0 else None
    tdma_vs_noc_l1_share = safe_div(_bundle, _bundle + _noc_out + _noc_in) if _l1_0 else None
    # Contention index: mean back-pressure over the 5 primary request/grant port pairs.
    _CONTENTION = ("L1_0_UNPACKER_0",) + _R0_OUT + _R0_IN
    _c_bps = [
        bounded(one_minus(safe_div(v.count("L1", n + "_GRANT"), v.count("L1", n))))
        for n in _CONTENTION
        if v.has(n) and v.has(n + "_GRANT")
    ]
    _c_bps = [b for b in _c_bps if b is not None]
    l1_contention_index = (sum(_c_bps) / len(_c_bps)) if _c_bps else None
    # NoC-vs-compute balance: NoC ring0 traffic vs FPU work.
    _noc_total = _noc_out + _noc_in
    noc_vs_compute_balance = (
        safe_div(_noc_total, fpu_instruction + _noc_total) if _l1_0 and v.has("FPU_COUNTER") else None
    )

    # Stall-cause overlap (UNBOUNDED ratio): summed per-resource wait rate over instrn_cycles, NOT
    # over THREAD_STALLS_t (a narrower, incommensurate event); >1 means several waits overlap.
    def _stall_overlap(t):
        reasons = [
            f"WAITING_FOR_THCON_IDLE_{t}",
            f"WAITING_FOR_UNPACK_IDLE_{t}",
            f"WAITING_FOR_PACK_IDLE_{t}",
            f"WAITING_FOR_MATH_IDLE_{t}",
            f"WAITING_FOR_NONZERO_SEM_{t}",
            f"WAITING_FOR_NONFULL_SEM_{t}",
            f"WAITING_FOR_MOVE_IDLE_{t}",
            f"WAITING_FOR_CFG_IDLE_{t}",
            f"WAITING_FOR_SFPU_IDLE_{t}",
        ]
        present = [r for r in reasons if v.has(r)]
        return safe_div(sum(v.count("INSTRN_THREAD", r) for r in present), instrn_cycles) if present else None

    stall_overlap_t0 = _stall_overlap(0)
    stall_overlap_t1 = _stall_overlap(1)
    stall_overlap_t2 = _stall_overlap(2)

    # UNBOUNDED ratio: math/SFPU ops per unpacker-busy cycle; >1 = compute-bound, <1 = unpack-bound.
    compute_to_unpack = (
        safe_div(fpu_or_sfpu, unpack0_busy + unpack1_busy)
        if v.has("MATH_COUNTER") and v.has("UNPACK0_BUSY_THREAD0")
        else None
    )

    _ring1_req = sum(v.count("L1", c) for c in L1_RING1 if v.has(c))
    _ring1_grant = sum(v.count("L1", c + "_GRANT") for c in L1_RING1 if v.has(c + "_GRANT"))
    noc_ring1_grant_eff = bounded(safe_div(_ring1_grant, _ring1_req))

    any_thread_stall = _instrn_rate("ANY_THREAD_STALL")

    l1_unpacker1_ext_backpressure = _bp(L1_UNPACKER1_EXT)
    l1_unpacker0_ext_backpressure = _bp(L1_UNPACKER0_EXT)
    l1_ext_pack_backpressure = _bp(L1_EXT_PACK)
    l1_tag_search_backpressure = _bp(L1_TAG_SEARCH)

    # Per-thread unpacker / src-write shares (fraction driven by each thread), bounded x/(x+y).
    _u0_t0 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    _u0_t1 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD1")
    _u1_t0 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    _u1_t1 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD1")
    unpack0_thread1_share = safe_div(_u0_t1, _u0_t0 + _u0_t1) if v.has("UNPACK0_BUSY_THREAD1") else None
    unpack1_thread1_share = safe_div(_u1_t1, _u1_t0 + _u1_t1) if v.has("UNPACK1_BUSY_THREAD1") else None
    _sa_even = v.count("TDMA_UNPACK", "SRCA_WRITE_TID_EVEN")
    _sa_odd = v.count("TDMA_UNPACK", "SRCA_WRITE_TID_ODD")
    _sb_even = v.count("TDMA_UNPACK", "SRCB_WRITE_TID_EVEN")
    _sb_odd = v.count("TDMA_UNPACK", "SRCB_WRITE_TID_ODD")
    srca_write_even_share = safe_div(_sa_even, _sa_even + _sa_odd) if v.has("SRCA_WRITE_TID_ODD") else None
    srcb_write_even_share = safe_div(_sb_even, _sb_even + _sb_odd) if v.has("SRCB_WRITE_TID_ODD") else None

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
        "data_hazard_stall_pct": pct(data_hazard_stall),
        "math_scoreboard_stall_pct": pct(math_scoreboard_stall),
        "math_pipeline_util_pct": pct(math_pipeline_util),
        # L1 composite (mean utilization across all present L1 client ports)
        "l1_mean_client_util_pct": pct(l1_mean_client_util),
        # L1 / NoC utilization (mean per-port busy fraction within each client group)
        "noc_ring0_util_pct": pct(noc_ring0_util),
        "noc_ring1_util_pct": pct(noc_ring1_util),
        "noc_ring0_grant_eff_pct": pct(noc_ring0_grant_eff),
        "l1_unpacker_util_pct": pct(unpacker_l1_util),
        "l1_unpacker1_ext_util_pct": pct(unpacker1_ext_l1_util),
        "l1_unpacker0_ext_util_pct": pct(unpacker0_ext_l1_util),
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
        "cfg_idle_wait_t0_pct": pct(cfg_idle_wait_t0),
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
        "l1_port1_util_pct": pct(l1_port1_util),
        "l1_packer_port8_util_pct": pct(l1_packer_port8_util),
        "unpacker_l1_eff_ratio": unpacker_l1_eff,
        "packer_l1_eff_ratio": packer_l1_eff,
        "l1_unpacker_backpressure_pct": pct(l1_unpacker_backpressure),
        "l1_port1_backpressure_pct": pct(l1_port1_backpressure),
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
        "l1_unpacker1_ext_backpressure_pct": pct(l1_unpacker1_ext_backpressure),
        "l1_unpacker0_ext_backpressure_pct": pct(l1_unpacker0_ext_backpressure),
        "l1_ext_pack_backpressure_pct": pct(l1_ext_pack_backpressure),
        "l1_tag_search_backpressure_pct": pct(l1_tag_search_backpressure),
        "unpack0_thread1_share_pct": pct(unpack0_thread1_share),
        "unpack1_thread1_share_pct": pct(unpack1_thread1_share),
        "srca_write_even_tid_share_pct": pct(srca_write_even_share),
        "srcb_write_even_tid_share_pct": pct(srcb_write_even_share),
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
    "cfg_idle_wait_t0_pct": "CFG Idle Wait T0",
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
    "l1_port1_util_pct": "L1 Port 1 Util",
    "l1_packer_port8_util_pct": "L1 Packer Port 8 Util",
    "l1_tdma_bundle_util_pct": "L1 TDMA Bundle Util",
    "l1_unpacker1_ext_util_pct": "L1 Unpacker1 Ext Util",
    "l1_unpacker0_ext_util_pct": "L1 Unpacker0 Ext Util",
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
    "l1_port1_backpressure_pct": "L1 Port 1 Backpressure",
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
    "l1_unpacker1_ext_backpressure_pct": "L1 Unpacker1 Ext Backpressure",
    "l1_unpacker0_ext_backpressure_pct": "L1 Unpacker0 Ext Backpressure",
    "l1_ext_pack_backpressure_pct": "L1 Ext Packer Backpressure",
    "l1_tag_search_backpressure_pct": "L1 Tag Search Backpressure",
    "unpack0_thread1_share_pct": "Unpacker0 T1 Share",
    "unpack1_thread1_share_pct": "Unpacker1 T1 Share",
    "srca_write_even_tid_share_pct": "SrcA Write Even-TID Share",
    "srcb_write_even_tid_share_pct": "SrcB Write Even-TID Share",
}


# The two display families of the module docstring, for consumers that key by metric key or by label.
RATIO_KEYS = {k for k in METRIC_LABELS if k.endswith("_ratio")}
RATIO_LABELS = {METRIC_LABELS[k] for k in RATIO_KEYS}
