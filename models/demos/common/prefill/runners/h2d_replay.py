# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Replay a captured tt-d-gen H2D inject log through the prefill producer.

tt-d-gen's PrefillPipeline logs one ``prefill inject slot_id=S start=A end=B`` line per chunk it
pushes, stamped by its builtin sink with local wall-clock time. The log carries the push order, slot
ids, chunk bounds and timing, not the tokens: the producer refills every chunk from its trace pool
by absolute position, exactly as it does for a synthetic schedule.
"""

import re
from dataclasses import dataclass
from datetime import datetime

_INJECT_RE = re.compile(r"prefill inject slot_id=(\d+) start=(\d+) end=(\d+)")
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_SHUTDOWN_SLOT = 0xFFFFFFFF


@dataclass(frozen=True)
class InjectRecord:
    slot_id: int
    actual_start: int
    actual_end: int
    actual_isl: int
    chunk_idx: int
    t_s: float | None


def _parse_timestamp(line: str) -> float | None:
    match = _TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.fromisoformat(match.group(1)).timestamp()


def parse_inject_log(lines, *, chunk_size: int, max_seq_len: int, num_slots: int) -> list:
    """Return the log's pushes in order as InjectRecords.

    Consecutive pushes to one slot whose start is the previous end form one contiguous run. Every
    push in a run gets the run's final end as ``actual_isl``, so MTP lookahead reads the tokens the
    run really carried past each chunk, and ``chunk_idx`` counts from the start of the run.
    """
    raw = []
    for lineno, line in enumerate(lines, start=1):
        match = _INJECT_RE.search(line)
        if match is None:
            continue
        slot_id, start, end = (int(g) for g in match.groups())
        if slot_id == _SHUTDOWN_SLOT:
            continue
        if not 0 <= start < end <= min(start + chunk_size, max_seq_len):
            raise ValueError(
                f"line {lineno}: start={start} end={end} is not a chunk inside "
                f"chunk_size={chunk_size} max_seq_len={max_seq_len}"
            )
        if slot_id >= num_slots:
            raise ValueError(f"line {lineno}: slot_id={slot_id} >= num_users={num_slots}")
        raw.append((slot_id, start, end, _parse_timestamp(line)))

    run_of = [0] * len(raw)
    run_end: list = []
    open_run: dict = {}
    for i, (slot_id, start, end, _) in enumerate(raw):
        prev = open_run.get(slot_id)
        if prev is None or run_end[prev] != start:
            prev = len(run_end)
            run_end.append(end)
            open_run[slot_id] = prev
        else:
            run_end[prev] = end
        run_of[i] = prev

    records = []
    chunk_in_run: dict = {}
    for i, (slot_id, start, end, t_s) in enumerate(raw):
        run = run_of[i]
        chunk_idx = chunk_in_run.get(run, 0)
        chunk_in_run[run] = chunk_idx + 1
        records.append(InjectRecord(slot_id, start, end, run_end[run], chunk_idx, t_s))
    return records


def load_inject_log(path: str, *, chunk_size: int, max_seq_len: int, num_slots: int) -> list:
    with open(path, errors="replace") as f:
        records = parse_inject_log(f, chunk_size=chunk_size, max_seq_len=max_seq_len, num_slots=num_slots)
    if not records:
        raise ValueError(f"{path}: no 'prefill inject' lines")
    return records
