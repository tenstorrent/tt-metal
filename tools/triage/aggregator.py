#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-rank aggregation primitives. Pure data shapes — no MPI, no transport."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass
from typing import Any


class Sentinel:
    """Marker base for non-data parts (failures, timeouts, skipped)."""


def is_sentinel(part: Any) -> bool:
    return isinstance(part, Sentinel)


@dataclass
class IndexedRow:
    index: int
    row: Any


@dataclass
class MergedResult:
    rows: Any
    sentinels: list[Any] = field(default_factory=list)


def default_merge(parts: list[Any]) -> MergedResult:
    """Collect rows + sentinels from each part. Each row carries its origin index."""
    sentinels: list[Any] = []
    rows: list[IndexedRow] = []
    for index, part in enumerate(parts):
        if is_sentinel(part):
            sentinels.append(part)
            continue
        if part is None:
            continue
        if isinstance(part, list):
            for item in part:
                rows.append(IndexedRow(index=index, row=item))
        else:
            rows.append(IndexedRow(index=index, row=part))
    return MergedResult(rows=rows, sentinels=sentinels)


_TAGGED_TYPE_CACHE: dict[tuple[type, str, str], type] = {}


def _build_tagged_dataclass(wrapped_cls: type, tag_field_name: str, tag_column_header: str) -> type:
    cache_key = (wrapped_cls, tag_field_name, tag_column_header)
    cached = _TAGGED_TYPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Lazy: triage_script may indirectly load this module.
    from triage_script import triage_field

    fields_spec: list[tuple[str, type, Any]] = [
        (tag_field_name, int, triage_field(tag_column_header)),
    ]
    for fld in fields(wrapped_cls):
        fields_spec.append((fld.name, fld.type, field(metadata=dict(fld.metadata))))
    tagged_cls = make_dataclass(f"Tagged_{wrapped_cls.__name__}", fields_spec)
    _TAGGED_TYPE_CACHE[cache_key] = tagged_cls
    return tagged_cls


def merged_to_renderable(
    merged: Any,
    *,
    tag_field_name: str = "rank",
    tag_column_header: str = "Rank",
) -> Any:
    """Turn a `MergedResult` of `IndexedRow` dataclass rows into a list of
    synthesized tagged dataclasses with a leading tag column. Other shapes pass through."""
    if not isinstance(merged, MergedResult):
        return merged
    rows = merged.rows
    if not isinstance(rows, list) or not rows:
        return rows
    if not all(isinstance(item, IndexedRow) for item in rows):
        return rows
    wrapped_samples = [item.row for item in rows]
    if not all(is_dataclass(sample) for sample in wrapped_samples):
        return [item.row for item in rows]
    wrapped_cls = type(wrapped_samples[0])
    if not all(type(sample) is wrapped_cls for sample in wrapped_samples):
        # Mixed dataclass types — can't build a single tagged class; unwrap.
        return wrapped_samples
    tagged_cls = _build_tagged_dataclass(wrapped_cls, tag_field_name, tag_column_header)
    out = []
    for item in rows:
        kwargs = {fld.name: getattr(item.row, fld.name) for fld in fields(item.row)}
        kwargs[tag_field_name] = item.index
        out.append(tagged_cls(**kwargs))
    return out
