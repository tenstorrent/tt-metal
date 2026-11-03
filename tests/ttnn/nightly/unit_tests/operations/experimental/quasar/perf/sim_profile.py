# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Slice the device-profiler CSV that a simulator run appends to, and reduce one op's rows to per-RISC cycle metrics.

On Quasar the profiler is CSV-only, the log is opened in append mode, CHIP_FREQ is 0 on the simulator and the
run-host id stays 0 under slow dispatch, so the only reliable per-op key is "rows appended since the previous
read". Only cycle columns are used; the ns columns are derived from a hard-coded clock.
"""

import csv
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

KERNEL_ZONE_RE = re.compile(r"-KERNEL$")
FW_ZONE_RE = re.compile(r"-FW$")
# Capacity of the L1-only profiler buffer per RISC on Quasar; a RISC at the cap has dropped markers.
ZONE_CAP_PER_RISC = 125


@dataclass
class RiscStats:
    risc: str
    core: tuple
    zones: int = 0
    busy: int = 0
    first_start: int | None = None
    last_end: int | None = None


@dataclass
class OpProfile:
    span: int | None = None
    n_cores: int = 0
    riscs: dict = field(default_factory=dict)  # (core, risc) -> RiscStats
    overflow: list = field(default_factory=list)
    fw_span: int | None = None

    def by_risc_type(self):
        """Kernel lifetime per RISC label (QUASAR_DM2, QUASAR_NEO0_TRISC2, ...) as a fraction of the program span,
        averaged over the active cores. The KERNEL zone runs from kernel entry to exit, so blocking waits count:
        a value near 1 means the hart stayed alive to the end of the program, not that it was busy."""
        agg = defaultdict(lambda: [0, 0])
        for (core, risc), st in self.riscs.items():
            agg[risc][0] += st.busy
            agg[risc][1] += 1
        if not self.span:
            return {}
        return {r: (busy / n) / self.span for r, (busy, n) in agg.items()}

    def active_set(self):
        return sorted({risc for (_, risc) in self.riscs})


def read_new_rows(path, offset):
    """Return (rows, new_offset) for everything appended to the CSV after byte offset `offset`."""
    if not os.path.exists(path):
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    text = chunk.decode("utf-8", errors="replace")
    rows = []
    for line in io.StringIO(text):
        line = line.strip()
        if not line or line.startswith("ARCH:") or line.startswith("PCIe slot"):
            continue
        rows.append(next(csv.reader([line])))
    return rows, offset + len(chunk)


def reduce_rows(rows):
    """Pair ZONE_START/ZONE_END markers per (core, RISC, zone) and reduce to span and per-RISC busy cycles."""
    prof = OpProfile()
    open_zones = {}
    kernel_bounds = []
    fw_bounds = []
    for r in rows:
        if len(r) < 12:
            continue
        core = (int(r[1]), int(r[2]))
        risc = r[3].strip()
        ts = int(r[5])
        name = r[10].strip()
        kind = r[11].strip().upper()
        is_kernel = bool(KERNEL_ZONE_RE.search(name))
        is_fw = bool(FW_ZONE_RE.search(name))
        if not (is_kernel or is_fw):
            continue
        key = (core, risc, name)
        if "START" in kind:
            open_zones[key] = ts
        elif "END" in kind and key in open_zones:
            start = open_zones.pop(key)
            if is_fw:
                fw_bounds.append((start, ts))
                continue
            st = prof.riscs.setdefault((core, risc), RiscStats(risc, core))
            st.zones += 1
            st.busy += ts - start
            st.first_start = start if st.first_start is None else min(st.first_start, start)
            st.last_end = ts if st.last_end is None else max(st.last_end, ts)
            kernel_bounds.append((start, ts))
    if kernel_bounds:
        prof.span = max(e for _, e in kernel_bounds) - min(s for s, _ in kernel_bounds)
    if fw_bounds:
        prof.fw_span = max(e for _, e in fw_bounds) - min(s for s, _ in fw_bounds)
    prof.n_cores = len({core for (core, _) in prof.riscs})
    prof.overflow = [f"{c}:{r}" for (c, r), st in prof.riscs.items() if st.zones >= ZONE_CAP_PER_RISC - 5]
    return prof


def reduce_all_zones(rows):
    """Pair every ZONE_START/ZONE_END by (core, RISC, zone name) and return per-zone duration lists.

    Used for calibration kernels whose zones are not the firmware KERNEL/FW ones; nested zones with the same
    name on one RISC pair in stack order.
    """
    stacks = defaultdict(list)
    durations = defaultdict(list)
    for r in rows:
        if len(r) < 12:
            continue
        key = ((int(r[1]), int(r[2])), r[3].strip(), r[10].strip())
        kind = r[11].strip().upper()
        ts = int(r[5])
        if "START" in kind:
            stacks[key].append(ts)
        elif "END" in kind and stacks[key]:
            durations[key].append(ts - stacks[key].pop())
    return {k: v for k, v in durations.items()}


def summarize_zones(durations):
    out = []
    for (core, risc, name), ds in sorted(durations.items()):
        ds_sorted = sorted(ds)
        out.append(
            {
                "core": core,
                "risc": risc,
                "zone": name,
                "n": len(ds),
                "min": ds_sorted[0],
                "median": ds_sorted[len(ds) // 2],
                "max": ds_sorted[-1],
                "sum": sum(ds),
            }
        )
    return out
