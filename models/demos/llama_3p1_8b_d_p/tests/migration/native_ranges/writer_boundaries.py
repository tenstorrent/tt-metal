# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure contract for the fixed 2K BFP8 cache-writer boundary test."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

PAGE_TOKENS = 32
HEAD_DIM = 128
BF8_PAGE_BYTES = 4352
MAX_SEQ_LEN = 2048
NUM_LAYERS = 32
NUM_SLOTS = 2
NUM_HEADS = 8
CONFIG_NAMES = tuple([f"k_h{i}" for i in range(NUM_HEADS)] + [f"v_h{i}" for i in range(NUM_HEADS)])


@dataclass(frozen=True)
class BoundaryCase:
    layer: int
    slot: int
    start: int
    end: int


WRITER_CASES = (
    BoundaryCase(0, 0, 0, 31),
    BoundaryCase(1, 1, 0, 32),
    BoundaryCase(7, 0, 0, 33),
    BoundaryCase(8, 1, 224, 255),
    BoundaryCase(15, 0, 224, 256),
    BoundaryCase(16, 1, 224, 257),
    BoundaryCase(23, 0, 992, 1023),
    BoundaryCase(24, 1, 992, 1024),
    BoundaryCase(31, 0, 992, 1025),
    BoundaryCase(30, 1, 2016, 2047),
)


def run_with_tensor_cleanup(operation, named_tensors, cleanup_errors):
    """Run one writer call, then release every temporary tensor without masking its failure."""
    primary_error = None
    primary_traceback = None
    result = None
    try:
        result = operation()
    except BaseException as caught:
        primary_error = caught
        primary_traceback = caught.__traceback__

    release_errors = []
    for name, tensor in reversed(named_tensors):
        try:
            tensor.deallocate(True)
        except Exception as caught:
            release_errors.append(f"{name} deallocate: {type(caught).__name__}: {caught}")
    cleanup_errors.extend(release_errors)

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    if release_errors:
        raise RuntimeError("; ".join(release_errors))
    return result


class PageKey(NamedTuple):
    kind: str
    head: int
    layer: int
    slot: int
    position: int


@dataclass(frozen=True)
class PageRows:
    valid_positions: tuple[int, ...]
    padding_positions: tuple[int, ...]


@dataclass(frozen=True)
class SnapshotSpec:
    phase: str
    slot: int
    begin: int
    end: int
    path: Path

    @property
    def expected_bytes(self) -> int:
        positions = (self.end - self.begin) // PAGE_TOKENS
        return len(CONFIG_NAMES) * NUM_LAYERS * positions * BF8_PAGE_BYTES


def device_major_positions(start: int) -> tuple[int, ...]:
    if type(start) is not int or start % PAGE_TOKENS or not 0 <= start < MAX_SEQ_LEN:
        raise ValueError(f"start must be 32-aligned inside the 2K logical cache, got {start!r}")
    positions = range(start, start + 1024)
    grouped = tuple(position for owner in range(4) for position in positions if (position % 1024) // 256 == owner)
    if len(grouped) != 1024 or any(
        sum((position % 1024) // 256 == owner for position in grouped) != 256 for owner in range(4)
    ):
        raise ValueError(f"start {start} does not produce four complete SP shards")
    return grouped


def iter_all_keys():
    for config in CONFIG_NAMES:
        kind, head_text = config.split("_h")
        for layer in range(NUM_LAYERS):
            for slot in range(NUM_SLOTS):
                for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                    yield PageKey(kind, int(head_text), layer, slot, position)


def case_page_positions(case: BoundaryCase) -> tuple[int, ...]:
    first = case.start // PAGE_TOKENS * PAGE_TOKENS
    last = (case.end - 1) // PAGE_TOKENS * PAGE_TOKENS
    return tuple(range(first, last + PAGE_TOKENS, PAGE_TOKENS))


def touched_keys() -> frozenset[PageKey]:
    keys = {
        PageKey(kind, head, case.layer, case.slot, position)
        for case in WRITER_CASES
        for position in case_page_positions(case)
        for kind in ("k", "v")
        for head in range(NUM_HEADS)
    }
    return frozenset(keys)


def classify_page_rows(case: BoundaryCase, page_position: int) -> PageRows:
    if page_position not in case_page_positions(case):
        return PageRows((), ())
    positions = range(page_position, page_position + PAGE_TOKENS)
    valid = tuple(position for position in positions if case.start <= position < case.end)
    padding = tuple(position for position in positions if case.end <= position < ((case.end + 31) // 32) * 32)
    return PageRows(valid, padding)


def _tag_row(phase: str, kind: str, head: int, layer: int, slot: int, position: int) -> tuple[float, ...]:
    if phase not in ("seed", "write"):
        raise ValueError(f"unknown phase {phase}")
    if kind not in ("k", "v"):
        raise ValueError(f"unknown kind {kind}")
    if not 0 <= head < NUM_HEADS or not 0 <= layer < NUM_LAYERS or not 0 <= slot < NUM_SLOTS:
        raise ValueError("coordinate out of range")
    if not 0 <= position < MAX_SEQ_LEN:
        raise ValueError("position out of range")

    row = [32.0] * HEAD_DIM
    row[0] = 32.0 if kind == "k" else -32.0
    cursor = 1
    for value, bits in ((head, 3), (layer, 5), (slot, 1), (position, 11)):
        for bit in range(bits):
            row[cursor] = 64.0 if value & (1 << bit) else 32.0
            cursor += 1
    row[cursor] = 16.0 if phase == "seed" else -16.0
    row[cursor + 1] = -64.0 if (head + layer + slot + position + (phase == "write")) % 2 else 64.0
    return tuple(row)


def input_row(phase: str, kind: str, head: int, layer: int, slot: int, position: int) -> tuple[float, ...]:
    """Tag one physical writer-input row without treating beyond-capacity padding as cache position."""
    if type(position) is not int or not 0 <= position < MAX_SEQ_LEN + 1024 - PAGE_TOKENS:
        raise ValueError("physical input position is outside one padded 1K window")
    if position < MAX_SEQ_LEN:
        return _tag_row(phase, kind, head, layer, slot, position)

    # Validate all logical coordinates through the normal row contract, then encode
    # the full 12-bit physical position and an explicit poison marker. These rows
    # are input padding only; no cache oracle key is created beyond max_seq_len.
    _tag_row(phase, kind, head, layer, slot, 0)
    row = [32.0] * HEAD_DIM
    row[0] = 32.0 if kind == "k" else -32.0
    cursor = 1
    for value, bits in ((head, 3), (layer, 5), (slot, 1), (position, 12)):
        for bit in range(bits):
            row[cursor] = 64.0 if value & (1 << bit) else 32.0
            cursor += 1
    row[cursor] = 16.0 if phase == "seed" else -16.0
    row[cursor + 1] = -64.0 if (head + layer + slot + position + (phase == "write")) % 2 else 64.0
    row[cursor + 2] = -32.0
    return tuple(row)


def seed_row(kind: str, head: int, layer: int, slot: int, position: int) -> tuple[float, ...]:
    return _tag_row("seed", kind, head, layer, slot, position)


def write_row(kind: str, head: int, layer: int, slot: int, position: int) -> tuple[float, ...]:
    return _tag_row("write", kind, head, layer, slot, position)


_ZERO_ROW = (0.0,) * HEAD_DIM


def expected_page(key: PageKey, case: BoundaryCase | None) -> tuple[tuple[float, ...], ...]:
    rows = []
    page_is_touched = (
        case is not None
        and key.layer == case.layer
        and key.slot == case.slot
        and key.position in case_page_positions(case)
    )
    rounded_end = ((case.end + PAGE_TOKENS - 1) // PAGE_TOKENS) * PAGE_TOKENS if case else 0
    for position in range(key.position, key.position + PAGE_TOKENS):
        if page_is_touched and case.start <= position < case.end:
            rows.append(write_row(key.kind, key.head, key.layer, key.slot, position))
        elif page_is_touched and case.end <= position < rounded_end:
            rows.append(_ZERO_ROW)
        else:
            rows.append(seed_row(key.kind, key.head, key.layer, key.slot, position))
    return tuple(rows)


def case_for_key(key: PageKey) -> BoundaryCase | None:
    matches = [
        case
        for case in WRITER_CASES
        if case.layer == key.layer and case.slot == key.slot and key.position in case_page_positions(case)
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping writer cases for {key}")
    return matches[0] if matches else None


def snapshot_specs(directory: Path) -> tuple[SnapshotSpec, ...]:
    return tuple(
        SnapshotSpec(phase, slot, 0, MAX_SEQ_LEN, directory / f"{phase}-slot{slot}.pages")
        for phase in ("before", "after")
        for slot in range(NUM_SLOTS)
    )


def decoder_function_source(path: Path, expected_digest: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_digest:
        raise ValueError(f"decoder source digest mismatch: expected {expected_digest}, got {actual}")
    module = ast.parse(path.read_text(), filename=str(path))
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_bfp8_chunk"]
    if len(matches) != 1:
        raise ValueError("expected exactly one _decode_bfp8_chunk")
    source = ast.get_source_segment(path.read_text(), matches[0])
    if source is None:
        raise ValueError("could not extract _decode_bfp8_chunk")
    return source
