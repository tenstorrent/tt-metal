# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's NoC router counter markers (ids 9120 to 9136).
Port names, capture order, NoC count and what one count is come from tools/include/perf_counters/router_counters_*.h
through headers.py; the totals are in router_metrics.py.

One launch is a header marker (id 9120: noc count << 32 | option mask) and one marker per pair of values (id 9121 + k:
value[2k + 1] << 32 | value[2k]). The values are per-launch deltas modulo 2^32, NoC 0 first, then the selected banks in
table order, and inside a bank either one value (the selected virtual channels added up) or one value per selected
virtual channel when the option keeps them apart. A last pair with an odd value count carries 0 in its high half.
All markers of a launch carry the same timestamp.
"""

from typing import Dict, List, NamedTuple, Optional

from . import headers, router_metrics

HEADER_MARKER_ID = 9120
PAIR_MARKER_ID_BASE = 9121
MAX_PAIRS = 16
MARKER_IDS = tuple(range(HEADER_MARKER_ID, PAIR_MARKER_ID_BASE + MAX_PAIRS))

_MASK32 = 0xFFFFFFFF


class Launch(NamedTuple):
    option: int
    num_nocs: int
    values: List[
        int
    ]  # flat, in capture order; an odd count keeps the padding zero of the last pair


def capture_core(arch, risc_type) -> bool:
    """True for the one core per tile that captures: BRISC on Blackhole, DM0 on Quasar."""
    arch = headers.normalize_arch(arch)
    risc = str(risc_type).upper()
    if arch == "quasar":
        return risc == "QUASAR_DM0"
    return arch == "blackhole" and risc == "BRISC"


def unpack_launch(header: int, pairs: Dict[int, int]) -> Launch:
    """Rebuild one launch from the header payload and {pair index: payload}; raises when a pair is missing."""
    option, num_nocs = header & _MASK32, header >> 32
    flat: List[int] = []
    for k in range(len(pairs)):
        if k not in pairs:
            raise ValueError(f"pair {k} missing from the launch")
        flat += [pairs[k] & _MASK32, pairs[k] >> 32]
    return Launch(option, num_nocs, flat)


def value_labels(arch, option, include_dir=None) -> List[str]:
    """Names of the values of one NoC in capture order: the port, plus _VC<n> when the channels are kept apart."""
    selected = headers.router_option(arch, option, include_dir)
    if not selected.banks:
        raise ValueError(f"option {option:#x} selects no router bank on {arch}")
    if not selected.vcs:
        raise ValueError(f"option {option:#x} selects no virtual channel")
    if selected.per_vc:
        return [f"{b.port}_VC{vc}" for b in selected.banks for vc in selected.vcs]
    return [b.port for b in selected.banks]


def decode_launch(arch, launch: Launch, include_dir=None) -> Dict[int, Dict[str, int]]:
    """NoC index -> {value label: count over the launch}. Raises when the value count does not fit the option."""
    labels = value_labels(arch, launch.option, include_dir)
    expected = len(labels) * launch.num_nocs
    if len(launch.values) != expected + expected % 2 or any(launch.values[expected:]):
        raise ValueError(
            f"{len(launch.values)} values captured, option {launch.option:#x} on {arch} needs {expected}"
        )
    per_noc = len(labels)
    return {
        noc: {label: launch.values[noc * per_noc + i] for i, label in enumerate(labels)}
        for noc in range(launch.num_nocs)
    }


def decode_router_launch(arch, launch: Launch, include_dir=None) -> Dict[str, object]:
    """Named view of one launch: counts per NoC plus the totals and what a count is."""
    table = headers.router_counter_table(arch, include_dir)
    counts = decode_launch(arch, launch, include_dir)
    return {
        "option": launch.option,
        "count_kind": table.count_kind,
        "counts": counts,
        "metrics": {
            noc: router_metrics.compute_router_metrics(c) for noc, c in counts.items()
        },
    }


def group_launches(rows) -> Optional[Dict[tuple, Dict[int, int]]]:
    """(run_host_id, core_x, core_y, risc_type, record_time) -> {marker id: payload} for the router counter markers."""
    launches: Dict[tuple, Dict[int, int]] = {}
    for row in rows:
        marker_id = int(row["marker_id"])
        if marker_id not in MARKER_IDS:
            continue
        key = (
            row["run_host_id"],
            row["core_x"],
            row["core_y"],
            row["risc_type"],
            int(row["record_time"]),
        )
        launches.setdefault(key, {})[marker_id] = int(row["data"])
    return launches
