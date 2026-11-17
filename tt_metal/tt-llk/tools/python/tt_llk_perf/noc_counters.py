# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's NoC interface counter markers (ids 9100 to 9116).
Names, capture order, NoC count and data word size come from tools/include/perf_counters/noc_counters_*.h through
headers.py; the derived byte and request totals are in noc_metrics.py.

One launch is a header marker (id 9100: noc count << 32 | group mask) and one marker per pair of counters (id 9101 + k:
counter[2k + 1] << 32 | counter[2k]). The counters are the per-NoC deltas over the launch, NoC 0 first, in the
table order of the selected groups. All markers of a launch carry the same timestamp.
"""

from typing import Dict, List, NamedTuple, Optional

from . import headers, noc_metrics

HEADER_MARKER_ID = 9100
PAIR_MARKER_ID_BASE = 9101
MAX_PAIRS = 16
MARKER_IDS = tuple(range(HEADER_MARKER_ID, PAIR_MARKER_ID_BASE + MAX_PAIRS))

_MASK32 = 0xFFFFFFFF


class Launch(NamedTuple):
    group_mask: int
    num_nocs: int
    counters: List[int]  # flat: NoC 0 counters in table order, then NoC 1


def capture_core(arch, risc_type) -> bool:
    """True for the one core per tile that captures: BRISC on Wormhole and Blackhole, DM0 on Quasar."""
    arch = headers.normalize_arch(arch)
    risc = str(risc_type).upper()
    if arch == "quasar":
        return risc == "QUASAR_DM0"
    return arch in ("blackhole", "wormhole", "wormhole_b0") and risc == "BRISC"


def unpack_launch(header: int, pairs: Dict[int, int]) -> Launch:
    """Rebuild one launch from the header payload and {pair index: payload}; raises when a pair is missing."""
    group_mask, num_nocs = header & _MASK32, header >> 32
    flat: List[int] = []
    for k in range(len(pairs)):
        if k not in pairs:
            raise ValueError(f"pair {k} missing from the launch")
        flat += [pairs[k] & _MASK32, pairs[k] >> 32]
    return Launch(group_mask, num_nocs, flat)


def decode_launch(arch, launch: Launch, include_dir=None) -> Dict[int, Dict[str, int]]:
    """NoC index -> {counter name: count over the launch}. Raises when the pair count does not fit the tables."""
    selected = headers.noc_selected_counters(arch, launch.group_mask, include_dir)
    expected = len(selected) * launch.num_nocs
    if expected != len(launch.counters):
        raise ValueError(
            f"{len(launch.counters)} counters captured, group mask {launch.group_mask:#x} on {arch} needs {expected}"
        )
    per_noc = len(selected)
    return {
        noc: {
            entry.name: launch.counters[noc * per_noc + i]
            for i, entry in enumerate(selected)
        }
        for noc in range(launch.num_nocs)
    }


def decode_noc_launch(arch, launch: Launch, include_dir=None) -> Dict[str, object]:
    """Named view of one launch: counts per NoC plus the derived byte and request totals per NoC."""
    word_bytes = headers.noc_counter_table(arch, include_dir).word_bytes
    counts = decode_launch(arch, launch, include_dir)
    return {
        "group_mask": launch.group_mask,
        "word_bytes": word_bytes,
        "counts": counts,
        "metrics": {
            noc: noc_metrics.compute_noc_metrics(c, word_bytes)
            for noc, c in counts.items()
        },
    }


def group_launches(rows) -> Optional[Dict[tuple, Dict[int, Dict[int, int]]]]:
    """(run_host_id, core_x, core_y, risc_type, record_time) -> {marker id: payload} for the NoC counter markers."""
    launches: Dict[tuple, Dict[int, Dict[int, int]]] = {}
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
