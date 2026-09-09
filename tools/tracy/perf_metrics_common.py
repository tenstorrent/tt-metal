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
    """Clamp a fraction to [0, 1]. The L1 grant counter is the arbiter accept for its port, so grant never exceeds
    request on Blackhole (0 of 924 port/op pairs in the selector sweep); the clamp stays as a guard for Wormhole,
    which was not measured, and for the scoreboard stall whose two counters start in separate groups."""
    return None if value is None else min(1.0, max(0.0, value))


def strict(v: "CounterView", *names) -> bool:
    """True only when every named counter was captured, so a same-bank fraction reads N/A rather than 0 or 100."""
    return all(v.has(n) for n in names)


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
# Packer L1 interfaces on banks 3-4 (ports 26-34); port 34 is packer interface 1, arbitrated with the tag-search
# accelerator that tt-metal never enables.
L1_EXT_PACK = (
    tuple(f"L1_3_EXT_PACKER_{i}" for i in (2, 3, 4, 5))
    + tuple(f"L1_4_EXT_PACKER_{i}" for i in (6, 7))
    + ("L1_4_PACKER_IF_1_TAG_SEARCH",)
)
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
    + L1_TDMA_BUNDLE
)


# ── Quasar (A0): four threads, the INSTISSUE class, backend stall reasons OR-reduced across threads ──
# Metric key stem -> INSTRN counter, one per thread-ORed stall reason.
STALL_REASON_COUNTERS = {
    "tile_counter_stall_pack": "TILE_COUNTER_STALL_PACK",
    "tile_counter_stall_unpack": "TILE_COUNTER_STALL_UNPACK",
    "srcs_stall_pack": "SRCS_STALL_PACK",
    "srcs_stall_sfpu": "SRCS_STALL_SFPU",
    "srcs_stall_unpack": "SRCS_STALL_UNPACK",
    "dest_stall_pack": "DEST_STALL_PACK",
    "dest_stall_sfpu": "DEST_STALL_SFPU",
    "dest_stall_math": "DEST_STALL_MATH",
    "dest_stall_unpack": "DEST_STALL_UNPACK",
    "sfpu_data_hazard_stall": "SFPU_DATA_HAZARD_STALL",
    "fpu_data_hazard_stall": "FPU_DATA_HAZARD_STALL",
    "srcb_stall_unpack": "SRCB_STALL_UNPACK",
    "srca_stall_unpack": "SRCA_STALL_UNPACK",
    "dvalid_stall_math": "DVALID_STALL_MATH",
    "srca_stall_math": "SRCA_STALL_MATH",
}


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
    fpu_utilization = safe_div(fpu_instruction, fpu_cycles) if v.has("FPU_COUNTER") else None
    compute_utilization = safe_div(fpu_or_sfpu, fpu_cycles) if v.has("MATH_COUNTER") else None

    unpack_thread_stall = _instrn_rate("THREAD_STALLS_0")
    math_thread_stall = _instrn_rate("THREAD_STALLS_1")
    pack_thread_stall = _instrn_rate("THREAD_STALLS_2")

    math_sem_wait = _instrn_rate("WAITING_FOR_NONZERO_SEM_1")
    pack_sem_wait = _instrn_rate("WAITING_FOR_NONZERO_SEM_2")

    srca_write = v.count("TDMA_UNPACK", "SRCA_WRITE_NOT_BLOCKED_PORT")
    srcb_write = v.count("TDMA_UNPACK", "SRCB_WRITE_NOT_BLOCKED_OVR")
    unpack0_busy = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    unpack1_busy = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")

    # UNBOUNDED ratios: source writes per unpacker busy cycle. SRCA_WRITE_REQ also counts THCON and thread-1
    # writes, so it is not a subset of UNPACK0_BUSY_THREAD0; above 1 means writes from outside the unpacker.
    srca_avail = v.count("TDMA_UNPACK", "SRCA_WRITE_REQ")
    srcb_avail = v.count("TDMA_UNPACK", "SRCB_WRITE_REQ")
    flow0 = safe_div(srca_avail, unpack0_busy) if strict(v, "SRCA_WRITE_REQ", "UNPACK0_BUSY_THREAD0") else None
    flow1 = safe_div(srcb_avail, unpack1_busy) if strict(v, "SRCB_WRITE_REQ", "UNPACK1_BUSY_THREAD0") else None
    flow_avg = avg_pair(flow0, flow1)

    # Packer Metrics: aggregate IDs work on both WH (per-engine also exposed) and BH (single packer).
    packer_busy = v.count("TDMA_PACK", "PACKER_BUSY")
    pack_utilization = safe_div(packer_busy, pack_cycles) if v.has("PACKER_BUSY") else None
    dest_read = v.count("TDMA_PACK", "PACKER0_DEST_READ_REQ")
    # A dest read request implies a non-empty packer request FIFO, which is the packer busy condition.
    pack_dest_eff = safe_div(dest_read, packer_busy) if strict(v, "PACKER0_DEST_READ_REQ", "PACKER_BUSY") else None

    math_available = v.count("TDMA_UNPACK", "MATH_INSTRN_AVAILABLE")
    # No src-data stall metric: MATH_SRC_DATA_READY is gated on dec_instr_alu while
    # MATH_INSTRN_AVAILABLE counts the whole math pipe, so their ratio is not a stall fraction.

    noc_ring0_util = mean_port_util(v, "L1", L1_RING0, l1_cycles)
    noc_ring1_util = mean_port_util(v, "L1", L1_RING1, l1_cycles)
    unpacker_l1_util = mean_port_util(v, "L1", L1_UNPACKER, l1_cycles)
    unpacker1_ext_l1_util = mean_port_util(v, "L1", L1_UNPACKER1_EXT, l1_cycles)
    unpacker0_ext_l1_util = mean_port_util(v, "L1", L1_UNPACKER0_EXT, l1_cycles)
    ext_pack_l1_util = mean_port_util(v, "L1", L1_EXT_PACK, l1_cycles)
    tdma_bundle_l1_util = mean_port_util(v, "L1", L1_TDMA_BUNDLE, l1_cycles)
    l1_mean_client_util = mean_port_util(v, "L1", L1_ALL, l1_cycles)
    # NoC ring0 grant efficiency: accepted per requested cycle, summed over the ports that carry both counters.
    _ring0_pairs = [c for c in L1_RING0 if strict(v, c, c + "_GRANT")]
    _ring0_req = sum(v.count("L1", c) for c in _ring0_pairs)
    _ring0_grant = sum(v.count("L1", c + "_GRANT") for c in _ring0_pairs)
    noc_ring0_grant_eff = bounded(safe_div(_ring0_grant, _ring0_req)) if _ring0_pairs else None

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
    # Idle engines count as zero (100% imbalance), so gate on presence of all four, not on activity.
    _engines = ("PACKER_BUSY_0", "PACKER_BUSY_1", "PACKER_BUSY_2", "PACKER_BUSY")
    packer_imbalance = safe_div(max(pb) - min(pb), max(pb)) if all(v.has(n) for n in _engines) else None
    dest_granted = v.count("TDMA_PACK", "DEST_READ_GRANTED_0")
    pack_dest_grant_eff = (
        safe_div(dest_granted, dest_read) if strict(v, "DEST_READ_GRANTED_0", "PACKER0_DEST_READ_REQ") else None
    )

    srca_write_eff = (
        safe_div(srca_write, srca_avail) if strict(v, "SRCA_WRITE_NOT_BLOCKED_PORT", "SRCA_WRITE_REQ") else None
    )
    srcb_write_eff = (
        safe_div(srcb_write, srcb_avail) if strict(v, "SRCB_WRITE_NOT_BLOCKED_OVR", "SRCB_WRITE_REQ") else None
    )

    # Stall rates are complements of "not stalled" counters. The scoreboard one is cross-bank (numerator in
    # TDMA_PACK, denominator in TDMA_UNPACK), so it is also clamped at 0.
    data_hazard_stall = (
        one_minus(safe_div(v.count("TDMA_UNPACK", "MATH_NOT_D2S_STALLED"), math_available))
        if strict(v, "MATH_NOT_D2S_STALLED", "MATH_INSTRN_AVAILABLE")
        else None
    )
    math_scoreboard_stall = (
        bounded(one_minus(safe_div(v.count("TDMA_PACK", "MATH_NOT_SCOREBOARD_STALLED"), math_available)))
        if strict(v, "MATH_NOT_SCOREBOARD_STALLED", "MATH_INSTRN_AVAILABLE")
        else None
    )
    math_pipeline_util = (
        safe_div(v.count("TDMA_UNPACK", "MATH_INSTRN_STARTED"), math_available)
        if strict(v, "MATH_INSTRN_STARTED", "MATH_INSTRN_AVAILABLE")
        else None
    )

    l1_port1 = first_present(v, L1_PORT1_NAMES)
    l1_port8 = first_present(v, L1_PORT8_NAMES)

    sfpu_util = safe_div(v.count("FPU", "SFPU_COUNTER"), fpu_cycles) if v.has("SFPU_COUNTER") else None
    # UNBOUNDED ratio: FPU busy cycles per cycle thread 1 had a math instruction ready. FPU_COUNTER counts
    # dequeues from any thread, so it is not a subset of the thread-1 availability.
    fpu_exec_eff = (
        safe_div(fpu_instruction, v.count("INSTRN_THREAD", "MATH_INSTRN_AVAILABLE_1"))
        if strict(v, "FPU_COUNTER", "MATH_INSTRN_AVAILABLE_1")
        else None
    )
    # UNBOUNDED ratio: available-math per busy packer (bank cycles when idle); >1 means the packer
    # is the handoff bottleneck.
    available_math = v.count("TDMA_PACK", "MATH_NOT_SCOREBOARD_STALLED")
    if not strict(v, "MATH_NOT_SCOREBOARD_STALLED", "PACKER_BUSY"):
        math_to_pack_handoff = None
    elif packer_busy > 0:
        math_to_pack_handoff = safe_div(available_math, packer_busy)
    else:
        math_to_pack_handoff = safe_div(available_math, pack_cycles)

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

    srca_write_ovr_blocked = (
        one_minus(safe_div(v.count("TDMA_UNPACK", "SRCA_WRITE_NOT_BLOCKED_OVR"), srca_avail))
        if strict(v, "SRCA_WRITE_NOT_BLOCKED_OVR", "SRCA_WRITE_REQ")
        else None
    )
    srcb_write_port_blocked = (
        one_minus(safe_div(v.count("TDMA_UNPACK", "SRCB_WRITE_NOT_BLOCKED_PORT"), srcb_avail))
        if strict(v, "SRCB_WRITE_NOT_BLOCKED_PORT", "SRCB_WRITE_REQ")
        else None
    )

    # Port 2 carries TDMA bundle 0 (mover, packer read, THCON) together with BRISC, TRISC0 and NCRISC.
    l1_port2_util = (
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
    # Back-pressure = 1 - accepted/requested; the grant is the arbiter accept, clamped only as a guard (bounded()).
    l1_unpacker_backpressure = (
        bounded(one_minus(safe_div(v.count("L1", "L1_0_UNPACKER_0_GRANT"), v.count("L1", "L1_0_UNPACKER_0"))))
        if strict(v, "L1_0_UNPACKER_0", "L1_0_UNPACKER_0_GRANT")
        else None
    )
    l1_port1_backpressure = (
        bounded(one_minus(safe_div(v.count("L1", L1_PORT1_GRANT[l1_port1]), v.count("L1", l1_port1))))
        if l1_port1 and v.has(L1_PORT1_GRANT[l1_port1])
        else None
    )

    def _bp(names):
        pairs = [n for n in names if strict(v, n, n + "_GRANT")]  # request and grant of the same port only
        if not pairs:
            return None
        req = sum(v.count("L1", n) for n in pairs)
        grant = sum(v.count("L1", n + "_GRANT") for n in pairs)
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

    _ring1_pairs = [c for c in L1_RING1 if strict(v, c, c + "_GRANT")]
    _ring1_req = sum(v.count("L1", c) for c in _ring1_pairs)
    _ring1_grant = sum(v.count("L1", c + "_GRANT") for c in _ring1_pairs)
    noc_ring1_grant_eff = bounded(safe_div(_ring1_grant, _ring1_req)) if _ring1_pairs else None

    any_thread_stall = _instrn_rate("ANY_THREAD_STALL")

    l1_unpacker1_ext_backpressure = _bp(L1_UNPACKER1_EXT)
    l1_unpacker0_ext_backpressure = _bp(L1_UNPACKER0_EXT)
    l1_ext_pack_backpressure = _bp(L1_EXT_PACK)

    # Per-thread unpacker / src-write shares (fraction driven by each thread), bounded x/(x+y).
    _u0_t0 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD0")
    _u0_t1 = v.count("TDMA_UNPACK", "UNPACK0_BUSY_THREAD1")
    _u1_t0 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD0")
    _u1_t1 = v.count("TDMA_UNPACK", "UNPACK1_BUSY_THREAD1")
    unpack0_thread1_share = (
        safe_div(_u0_t1, _u0_t0 + _u0_t1) if strict(v, "UNPACK0_BUSY_THREAD0", "UNPACK0_BUSY_THREAD1") else None
    )
    unpack1_thread1_share = (
        safe_div(_u1_t1, _u1_t0 + _u1_t1) if strict(v, "UNPACK1_BUSY_THREAD0", "UNPACK1_BUSY_THREAD1") else None
    )
    _sa_even = v.count("TDMA_UNPACK", "SRCA_WRITE_TID_EVEN")
    _sa_odd = v.count("TDMA_UNPACK", "SRCA_WRITE_TID_ODD")
    _sb_even = v.count("TDMA_UNPACK", "SRCB_WRITE_TID_EVEN")
    _sb_odd = v.count("TDMA_UNPACK", "SRCB_WRITE_TID_ODD")
    srca_write_even_share = (
        safe_div(_sa_even, _sa_even + _sa_odd) if strict(v, "SRCA_WRITE_TID_EVEN", "SRCA_WRITE_TID_ODD") else None
    )
    srcb_write_even_share = (
        safe_div(_sb_even, _sb_even + _sb_odd) if strict(v, "SRCB_WRITE_TID_EVEN", "SRCB_WRITE_TID_ODD") else None
    )

    # ── Quasar (A0), all has()-gated so a tt-1xx capture reads None ──
    unpack_cycles = v.cycles("TDMA_UNPACK")

    def _gated_rate(bank, name, cycles):
        return safe_div(v.count(bank, name), cycles) if v.has(name) else None

    thread3_stall = _gated_rate("INSTRN_THREAD", "THREAD_STALLS_3", instrn_cycles)
    thread3_ipc = _gated_rate("INSTRN_THREAD", "THREAD_INSTRUCTIONS_3", instrn_cycles)

    def _avail(cls, t):
        return _gated_rate("INSTRN_THREAD", f"{cls}_INSTRN_AVAILABLE_{t}", instrn_cycles)

    # A stall reason's rate is over the INSTRN bank's cycles; its share is its part of every reason
    # captured, so it needs at least two of them to mean anything. DVALID_STALL_MATH is srcA-or-srcB not
    # valid and contains SRCA_STALL_MATH, so the share basis carries the derived srcB part instead of it.
    _reason_counts = {c: v.count("INSTRN_THREAD", c) for c in STALL_REASON_COUNTERS.values() if v.has(c)}
    _srcb_stall_math = (
        max(0.0, _reason_counts["DVALID_STALL_MATH"] - _reason_counts["SRCA_STALL_MATH"])
        if strict(v, "DVALID_STALL_MATH", "SRCA_STALL_MATH")
        else None
    )
    _share_basis = {c: n for c, n in _reason_counts.items() if c != "DVALID_STALL_MATH"}
    if _srcb_stall_math is not None:
        _share_basis["SRCB_STALL_MATH"] = _srcb_stall_math
    _reason_total = sum(_share_basis.values())

    def _reason_rate(c):
        return _gated_rate("INSTRN_THREAD", c, instrn_cycles)

    def _reason_share(c):
        if c not in _share_basis or len(_share_basis) < 2:
            return None
        return safe_div(_share_basis[c], _reason_total)

    srcb_stall_math = safe_div(_srcb_stall_math, instrn_cycles) if _srcb_stall_math is not None else None

    def _unpack_busy(u, t):
        return _gated_rate("TDMA_UNPACK", f"UNPACK{u}_BUSY_THREAD{t}", unpack_cycles)

    math_src_data_ready = _gated_rate("TDMA_UNPACK", "MATH_SRC_DATA_READY", unpack_cycles)
    # MATH_COUNTER counts fpu-or-sfpu cycles, so the cycles both units were busy are FPU + SFPU - MATH.
    fpu_sfpu_overlap = (
        safe_div(max(0.0, fpu_instruction + v.count("FPU", "SFPU_COUNTER") - fpu_or_sfpu), fpu_cycles)
        if all(v.has(n) for n in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"))
        else None
    )

    # UNBOUNDED ratio: instructions issued per cycle the thread was not stalled.
    def _per_ready_cycle(t):
        instr, stall = f"THREAD_INSTRUCTIONS_{t}", f"THREAD_STALLS_{t}"
        if not (v.has(instr) and v.has(stall)) or instrn_cycles <= 0:
            return None
        return safe_div(v.count("INSTRN_THREAD", instr), max(1.0, instrn_cycles - v.count("INSTRN_THREAD", stall)))

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
        # Unpacker-to-math flow (ratios: source writes per unpacker busy cycle)
        "unpack_to_math_flow0_ratio": flow0,
        "unpack_to_math_flow1_ratio": flow1,
        "unpack_to_math_flow_ratio": flow_avg,
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
        "packer_load_imbalance_pct": pct(packer_imbalance),
        "pack_dest_grant_eff_pct": pct(pack_dest_grant_eff),
        # Source-register write completion efficiency
        "srca_write_eff_pct": pct(srca_write_eff),
        "srcb_write_eff_pct": pct(srcb_write_eff),
        # Compute
        "sfpu_utilization_pct": pct(sfpu_util),
        "fpu_exec_eff_ratio": fpu_exec_eff,
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
        # Write-blocked rates (the other blocking mode of each source; the first is 1 - its efficiency)
        "srca_write_ovr_blocked_pct": pct(srca_write_ovr_blocked),
        "srcb_write_port_blocked_pct": pct(srcb_write_port_blocked),
        # L1 per-port + grant efficiency
        "l1_port2_util_pct": pct(l1_port2_util),
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
        "unpack0_thread1_share_pct": pct(unpack0_thread1_share),
        "unpack1_thread1_share_pct": pct(unpack1_thread1_share),
        "srca_write_even_tid_share_pct": pct(srca_write_even_share),
        "srcb_write_even_tid_share_pct": pct(srcb_write_even_share),
        # ── Quasar (A0) ──
        # Thread 3
        "thread3_stall_pct": pct(thread3_stall),
        "thread3_ipc_pct": pct(thread3_ipc),
        # Per-class instruction availability, the (class, thread) pairs not covered above
        "cfg_instrn_avail_t1_pct": pct(_avail("CFG", 1)),
        "cfg_instrn_avail_t2_pct": pct(_avail("CFG", 2)),
        "cfg_instrn_avail_t3_pct": pct(_avail("CFG", 3)),
        "sync_instrn_avail_t1_pct": pct(_avail("SYNC", 1)),
        "sync_instrn_avail_t2_pct": pct(_avail("SYNC", 2)),
        "sync_instrn_avail_t3_pct": pct(_avail("SYNC", 3)),
        "thcon_instrn_avail_t1_pct": pct(_avail("THCON", 1)),
        "thcon_instrn_avail_t2_pct": pct(_avail("THCON", 2)),
        "thcon_instrn_avail_t3_pct": pct(_avail("THCON", 3)),
        "instissue_instrn_avail_t0_pct": pct(_avail("INSTISSUE", 0)),
        "instissue_instrn_avail_t1_pct": pct(_avail("INSTISSUE", 1)),
        "instissue_instrn_avail_t2_pct": pct(_avail("INSTISSUE", 2)),
        "instissue_instrn_avail_t3_pct": pct(_avail("INSTISSUE", 3)),
        "math_instrn_avail_t0_pct": pct(_avail("MATH", 0)),
        "math_instrn_avail_t2_pct": pct(_avail("MATH", 2)),
        "math_instrn_avail_t3_pct": pct(_avail("MATH", 3)),
        "unpack_instrn_avail_t1_pct": pct(_avail("UNPACK", 1)),
        "unpack_instrn_avail_t2_pct": pct(_avail("UNPACK", 2)),
        "unpack_instrn_avail_t3_pct": pct(_avail("UNPACK", 3)),
        "pack_instrn_avail_t0_pct": pct(_avail("PACK", 0)),
        "pack_instrn_avail_t1_pct": pct(_avail("PACK", 1)),
        "pack_instrn_avail_t3_pct": pct(_avail("PACK", 3)),
        # Thread-ORed stall reasons: rate over cycles, then share of all captured reasons
        "tile_counter_stall_pack_pct": pct(_reason_rate("TILE_COUNTER_STALL_PACK")),
        "tile_counter_stall_unpack_pct": pct(_reason_rate("TILE_COUNTER_STALL_UNPACK")),
        "srcs_stall_pack_pct": pct(_reason_rate("SRCS_STALL_PACK")),
        "srcs_stall_sfpu_pct": pct(_reason_rate("SRCS_STALL_SFPU")),
        "srcs_stall_unpack_pct": pct(_reason_rate("SRCS_STALL_UNPACK")),
        "dest_stall_pack_pct": pct(_reason_rate("DEST_STALL_PACK")),
        "dest_stall_sfpu_pct": pct(_reason_rate("DEST_STALL_SFPU")),
        "dest_stall_math_pct": pct(_reason_rate("DEST_STALL_MATH")),
        "dest_stall_unpack_pct": pct(_reason_rate("DEST_STALL_UNPACK")),
        "sfpu_data_hazard_stall_pct": pct(_reason_rate("SFPU_DATA_HAZARD_STALL")),
        "fpu_data_hazard_stall_pct": pct(_reason_rate("FPU_DATA_HAZARD_STALL")),
        "srcb_stall_unpack_pct": pct(_reason_rate("SRCB_STALL_UNPACK")),
        "srca_stall_unpack_pct": pct(_reason_rate("SRCA_STALL_UNPACK")),
        "dvalid_stall_math_pct": pct(_reason_rate("DVALID_STALL_MATH")),
        "srca_stall_math_pct": pct(_reason_rate("SRCA_STALL_MATH")),
        "srcb_stall_math_pct": pct(srcb_stall_math),
        "tile_counter_stall_pack_share_pct": pct(_reason_share("TILE_COUNTER_STALL_PACK")),
        "tile_counter_stall_unpack_share_pct": pct(_reason_share("TILE_COUNTER_STALL_UNPACK")),
        "srcs_stall_pack_share_pct": pct(_reason_share("SRCS_STALL_PACK")),
        "srcs_stall_sfpu_share_pct": pct(_reason_share("SRCS_STALL_SFPU")),
        "srcs_stall_unpack_share_pct": pct(_reason_share("SRCS_STALL_UNPACK")),
        "dest_stall_pack_share_pct": pct(_reason_share("DEST_STALL_PACK")),
        "dest_stall_sfpu_share_pct": pct(_reason_share("DEST_STALL_SFPU")),
        "dest_stall_math_share_pct": pct(_reason_share("DEST_STALL_MATH")),
        "dest_stall_unpack_share_pct": pct(_reason_share("DEST_STALL_UNPACK")),
        "sfpu_data_hazard_stall_share_pct": pct(_reason_share("SFPU_DATA_HAZARD_STALL")),
        "fpu_data_hazard_stall_share_pct": pct(_reason_share("FPU_DATA_HAZARD_STALL")),
        "srcb_stall_unpack_share_pct": pct(_reason_share("SRCB_STALL_UNPACK")),
        "srca_stall_unpack_share_pct": pct(_reason_share("SRCA_STALL_UNPACK")),
        "srca_stall_math_share_pct": pct(_reason_share("SRCA_STALL_MATH")),
        "srcb_stall_math_share_pct": pct(_reason_share("SRCB_STALL_MATH")),
        # Unpacker busy per unpacker and thread
        "unpack0_busy_t0_pct": pct(_unpack_busy(0, 0)),
        "unpack1_busy_t0_pct": pct(_unpack_busy(1, 0)),
        "unpack2_busy_t0_pct": pct(_unpack_busy(2, 0)),
        "unpack0_busy_t1_pct": pct(_unpack_busy(0, 1)),
        "unpack1_busy_t1_pct": pct(_unpack_busy(1, 1)),
        # Math source readiness, FPU/SFPU overlap
        "math_src_data_ready_pct": pct(math_src_data_ready),
        "fpu_sfpu_overlap_pct": pct(fpu_sfpu_overlap),
        # Instructions per issue-ready cycle, per thread
        "thread0_instrn_per_ready_cycle_ratio": _per_ready_cycle(0),
        "thread1_instrn_per_ready_cycle_ratio": _per_ready_cycle(1),
        "thread2_instrn_per_ready_cycle_ratio": _per_ready_cycle(2),
        "thread3_instrn_per_ready_cycle_ratio": _per_ready_cycle(3),
    }


# Human display name per metric key (the Tracy tool's historical names where they existed, so its CSV
# columns / summary stay stable). Consumers that print or write CSVs map keys through this.
METRIC_LABELS = {
    "fpu_utilization_pct": "FPU Util",
    "compute_utilization_pct": "MATH Util",
    "sfpu_utilization_pct": "SFPU Util",
    "fpu_exec_eff_ratio": "FPU Execution Efficiency",
    "pack_utilization_pct": "Packer Utilization",
    "pack_dest_eff_pct": "Packer Efficiency",
    "pack_dest_grant_eff_pct": "Pack Dest Grant Efficiency",
    "math_pipeline_util_pct": "Math Pipeline Utilization",
    "math_to_pack_handoff_ratio": "Math-to-Pack Handoff Efficiency",
    "unpack_to_math_flow_ratio": "Unpacker-to-Math Data Flow",
    "unpack_to_math_flow0_ratio": "Unpacker-to-Math Data Flow (srcA)",
    "unpack_to_math_flow1_ratio": "Unpacker-to-Math Data Flow (srcB)",
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
    "srca_write_ovr_blocked_pct": "SrcA Write Overwrite Blocked Rate",
    "srcb_write_port_blocked_pct": "SrcB Write Port Blocked Rate",
    "thread0_ipc_pct": "T0 Instrn Issue Rate",
    "thread1_ipc_pct": "T1 Instrn Issue Rate",
    "thread2_ipc_pct": "T2 Instrn Issue Rate",
    "packer0_util_pct": "Packer Engine 0 Util",
    "packer1_util_pct": "Packer Engine 1 Util",
    "packer2_util_pct": "Packer Engine 2 Util",
    "packer_load_imbalance_pct": "Packer Load Imbalance",
    "l1_unpacker_util_pct": "L1 Unpacker Port Util",
    "l1_port1_util_pct": "L1 Port 1 Util",
    "l1_packer_port8_util_pct": "L1 Packer Port 8 Util",
    "l1_tdma_bundle_util_pct": "L1 TDMA Bundle Util",
    "l1_unpacker1_ext_util_pct": "L1 Unpacker1 Ext Util",
    "l1_unpacker0_ext_util_pct": "L1 Unpacker0 Ext Util",
    "l1_ext_pack_util_pct": "L1 Packer Interfaces Util",
    "l1_mean_client_util_pct": "L1 Mean Client Util",
    "l1_port2_util_pct": "L1 Port 2 Util",
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
    "l1_ext_pack_backpressure_pct": "L1 Packer Interfaces Backpressure",
    "unpack0_thread1_share_pct": "Unpacker0 T1 Share",
    "unpack1_thread1_share_pct": "Unpacker1 T1 Share",
    "srca_write_even_tid_share_pct": "SrcA Write Even-TID Share",
    "srcb_write_even_tid_share_pct": "SrcB Write Even-TID Share",
    # ── Quasar (A0) ──
    "thread3_stall_pct": "Thread 3 Stall Rate",
    "thread3_ipc_pct": "T3 Instrn Issue Rate",
    "cfg_instrn_avail_t1_pct": "CFG Instrn Avail Rate T1",
    "cfg_instrn_avail_t2_pct": "CFG Instrn Avail Rate T2",
    "cfg_instrn_avail_t3_pct": "CFG Instrn Avail Rate T3",
    "sync_instrn_avail_t1_pct": "SYNC Instrn Avail Rate T1",
    "sync_instrn_avail_t2_pct": "SYNC Instrn Avail Rate T2",
    "sync_instrn_avail_t3_pct": "SYNC Instrn Avail Rate T3",
    "thcon_instrn_avail_t1_pct": "THCON Instrn Avail Rate T1",
    "thcon_instrn_avail_t2_pct": "THCON Instrn Avail Rate T2",
    "thcon_instrn_avail_t3_pct": "THCON Instrn Avail Rate T3",
    "instissue_instrn_avail_t0_pct": "INSTISSUE Instrn Avail Rate T0",
    "instissue_instrn_avail_t1_pct": "INSTISSUE Instrn Avail Rate T1",
    "instissue_instrn_avail_t2_pct": "INSTISSUE Instrn Avail Rate T2",
    "instissue_instrn_avail_t3_pct": "INSTISSUE Instrn Avail Rate T3",
    "math_instrn_avail_t0_pct": "MATH Instrn Avail Rate T0",
    "math_instrn_avail_t2_pct": "MATH Instrn Avail Rate T2",
    "math_instrn_avail_t3_pct": "MATH Instrn Avail Rate T3",
    "unpack_instrn_avail_t1_pct": "UNPACK Instrn Avail Rate T1",
    "unpack_instrn_avail_t2_pct": "UNPACK Instrn Avail Rate T2",
    "unpack_instrn_avail_t3_pct": "UNPACK Instrn Avail Rate T3",
    "pack_instrn_avail_t0_pct": "PACK Instrn Avail Rate T0",
    "pack_instrn_avail_t1_pct": "PACK Instrn Avail Rate T1",
    "pack_instrn_avail_t3_pct": "PACK Instrn Avail Rate T3",
    "tile_counter_stall_pack_pct": "Tile Counter Stall Pack Rate",
    "tile_counter_stall_unpack_pct": "Tile Counter Stall Unpack Rate",
    "srcs_stall_pack_pct": "Srcs Stall Pack Rate",
    "srcs_stall_sfpu_pct": "Srcs Stall SFPU Rate",
    "srcs_stall_unpack_pct": "Srcs Stall Unpack Rate",
    "dest_stall_pack_pct": "Dest Stall Pack Rate",
    "dest_stall_sfpu_pct": "Dest Stall SFPU Rate",
    "dest_stall_math_pct": "Dest Stall Math Rate",
    "dest_stall_unpack_pct": "Dest Stall Unpack Rate",
    "sfpu_data_hazard_stall_pct": "SFPU Data Hazard Stall Rate",
    "fpu_data_hazard_stall_pct": "FPU Data Hazard Stall Rate",
    "srcb_stall_unpack_pct": "SrcB Stall Unpack Rate",
    "srca_stall_unpack_pct": "SrcA Stall Unpack Rate",
    "dvalid_stall_math_pct": "Src Valid Stall Math Rate",
    "srca_stall_math_pct": "SrcA Stall Math Rate",
    "srcb_stall_math_pct": "SrcB Stall Math Rate",
    "tile_counter_stall_pack_share_pct": "Tile Counter Stall Pack Share",
    "tile_counter_stall_unpack_share_pct": "Tile Counter Stall Unpack Share",
    "srcs_stall_pack_share_pct": "Srcs Stall Pack Share",
    "srcs_stall_sfpu_share_pct": "Srcs Stall SFPU Share",
    "srcs_stall_unpack_share_pct": "Srcs Stall Unpack Share",
    "dest_stall_pack_share_pct": "Dest Stall Pack Share",
    "dest_stall_sfpu_share_pct": "Dest Stall SFPU Share",
    "dest_stall_math_share_pct": "Dest Stall Math Share",
    "dest_stall_unpack_share_pct": "Dest Stall Unpack Share",
    "sfpu_data_hazard_stall_share_pct": "SFPU Data Hazard Stall Share",
    "fpu_data_hazard_stall_share_pct": "FPU Data Hazard Stall Share",
    "srcb_stall_unpack_share_pct": "SrcB Stall Unpack Share",
    "srca_stall_unpack_share_pct": "SrcA Stall Unpack Share",
    "srca_stall_math_share_pct": "SrcA Stall Math Share",
    "srcb_stall_math_share_pct": "SrcB Stall Math Share",
    "unpack0_busy_t0_pct": "Unpacker0 Busy T0 Util",
    "unpack1_busy_t0_pct": "Unpacker1 Busy T0 Util",
    "unpack2_busy_t0_pct": "Unpacker2 Busy T0 Util",
    "unpack0_busy_t1_pct": "Unpacker0 Busy T1 Util",
    "unpack1_busy_t1_pct": "Unpacker1 Busy T1 Util",
    "math_src_data_ready_pct": "Math Src Data Ready Rate",
    "fpu_sfpu_overlap_pct": "FPU SFPU Overlap",
    "thread0_instrn_per_ready_cycle_ratio": "T0 Instrn Per Issue-Ready Cycle",
    "thread1_instrn_per_ready_cycle_ratio": "T1 Instrn Per Issue-Ready Cycle",
    "thread2_instrn_per_ready_cycle_ratio": "T2 Instrn Per Issue-Ready Cycle",
    "thread3_instrn_per_ready_cycle_ratio": "T3 Instrn Per Issue-Ready Cycle",
}


# The two display families of the module docstring, for consumers that key by metric key or by label.
RATIO_KEYS = {k for k in METRIC_LABELS if k.endswith("_ratio")}
RATIO_LABELS = {METRIC_LABELS[k] for k in RATIO_KEYS}


# ── Quasar l1_client event counter: one clear-on-read CSR behind a subport*8 + event mux, selected per run ──
# Records are named after the selection, so this metric family is dynamic (compute_l1_client_metrics, metric_label).
L1_CLIENT_PREFIX = "L1_CLIENT_"
QUASAR_L1_CLIENT_NUM_SUBPORTS = 37
# Verified against the A0 L1 RTL; events 2-6 are counter carries (one pulse per lane count or order
# depth), events 1 and 7 are per-cycle indicators.
QUASAR_L1_CLIENT_EVENT_NAMES = (
    "UNUSED",
    "SBANK_POP",
    "ISSUE_STALL_CARRY",
    "ISSUE_WORK_CARRY",
    "FLEX_STALL_CARRY",
    "FLEX_WORK_CARRY",
    "PENDING_REQS_CARRY",
    "ORDER_FIFO_ACTIVE",
)


# Events 1-3 are per SBank of the whole port (the RTL indexes them with the sub-port number modulo the SBank count).
QUASAR_L1_CLIENT_SBANK_EVENTS = (1, 2, 3)


def quasar_l1_client_selection_is_valid(sel) -> bool:
    """False for selections that cannot carry data: out of range, event 0 (unused, reads 0 in the RTL), and the THCON
    sub-port's events 1-3 (the TRISC port's SBank 0 counters, which sub-port 0 already exposes)."""
    sel = int(sel)
    if not 0 <= sel < QUASAR_L1_CLIENT_NUM_SUBPORTS * 8:
        return False
    subport, event = divmod(sel, 8)
    return event != 0 and not (subport == 4 and event in QUASAR_L1_CLIENT_SBANK_EVENTS)


def quasar_l1_client_label(sel) -> str:
    """Counter name for an l1_client selection (subport*8 + event). Subports (t6_l1_client_map.sv): 0-3 TRISC, 4 THCON,
    5-24 unpacker reads (3 unpackers x 2 interfaces x 4 lanes, unpacker 2 has interface 0 only), 25-36 packer writes
    (packer 0 interfaces 0-1, packer 1 interface 0). Events 1-3 name the SBank of the port instead of the sub-port.
    """
    sel = int(sel)
    if not quasar_l1_client_selection_is_valid(sel):
        return f"{L1_CLIENT_PREFIX}INVALID_{sel}"
    subport, event = divmod(sel, 8)
    per_sbank = event in QUASAR_L1_CLIENT_SBANK_EVENTS
    if subport < 4:
        port = f"TRISC_SBANK{subport}" if per_sbank else f"TRISC{subport}"
    elif subport == 4:
        port = "THCON"
    else:
        unit_name, base = ("UNPACK", 5) if subport < 25 else ("PACK", 25)
        unit, rest = divmod(subport - base, 8)
        interface, lane = divmod(rest, 4)
        port = f"{unit_name}{unit}_IF{interface}_" + (f"SBANK{lane}" if per_sbank else f"LANE{lane}")
    return f"{L1_CLIENT_PREFIX}{port}_{QUASAR_L1_CLIENT_EVENT_NAMES[event]}"


def l1_client_pending_reqs_divisor(counter_name: str) -> float:
    """Outstanding-request cycles per PENDING_REQS_CARRY pulse: 2^clog2(3 * RSP_BUF_D), 128 on packer 0's two
    interfaces (24-deep response buffer) and 64 on every other sub-port."""
    port = str(counter_name)[len(L1_CLIENT_PREFIX) :].split("_")
    return 128.0 if port[0] == "PACK0" and port[1] in ("IF0", "IF1") else 64.0


def l1_client_is_ratio(counter_name_or_label: str) -> bool:
    """The pending-request carry reports mean outstanding requests, an unbounded ratio; every other event is a rate."""
    return "PENDING_REQS_CARRY" in str(counter_name_or_label).upper()


def l1_client_metric_key(counter_name: str) -> str:
    """Metric key of an l1_client counter: lower-case name plus the family suffix (_ratio for the pending carry)."""
    return f"{counter_name.lower()}_ratio" if l1_client_is_ratio(counter_name) else f"{counter_name.lower()}_pct"


def l1_client_metric_label(counter_name: str) -> str:
    """Display name of an l1_client metric: the counter name plus ' Rate', or ' Mean Outstanding' for the pending carry."""
    return f"{counter_name} Mean Outstanding" if l1_client_is_ratio(counter_name) else f"{counter_name} Rate"


def is_ratio_label(label: str) -> bool:
    """Whether a display label belongs to the unbounded ratio family (static RATIO_LABELS or the pending-request carry)."""
    return label in RATIO_LABELS or (str(label).startswith(L1_CLIENT_PREFIX) and l1_client_is_ratio(label))


def metric_label(key: str) -> str:
    """Display name for a metric key: METRIC_LABELS, or the dynamic l1_client family, else the key itself."""
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    if key.startswith(L1_CLIENT_PREFIX.lower()):
        for suffix in ("_pct", "_ratio"):
            if key.endswith(suffix):
                return l1_client_metric_label(key[: -len(suffix)].upper())
    return key


def compute_l1_client_metrics(v: CounterView, counter_names) -> dict:
    """Per-run value of every l1_client counter present over the capture's wall-clock span (the CSR has no reference
    counter). SBANK_POP and ORDER_FIFO_ACTIVE are cycle indicators; the ISSUE/FLEX carries fire once per four lane
    events, so carry / cycles is the mean per-lane fraction (bounded); the pending-request carry times its divisor
    over cycles is the mean number of outstanding requests (a ratio)."""
    cycles = v.cycles("L1_CLIENT")
    out = {}
    for name in sorted(set(counter_names)):
        if not str(name).startswith(L1_CLIENT_PREFIX) or not v.has(name):
            continue
        rate = safe_div(v.count("L1_CLIENT", name), cycles)
        if rate is None:
            out[l1_client_metric_key(name)] = None
        elif l1_client_is_ratio(name):
            out[l1_client_metric_key(name)] = rate * l1_client_pending_reqs_divisor(name)
        else:
            out[l1_client_metric_key(name)] = pct(rate)
    return out
