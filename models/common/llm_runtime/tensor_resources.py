# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Low-level helpers for releasing runtime-owned TT tensor trees."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Iterable

import ttnn


@dataclass
class TensorResourceOrphan:
    """An owned tensor tree whose failed releases remain retryable."""

    values: Any
    deallocated_tensor_ids: set[int] = field(default_factory=set)


def trim_host_allocator() -> None:
    """Return released trace-capture staging arenas to the OS when supported."""

    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = (ctypes.c_size_t,)
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def deallocate_owned_tensors(value: Any, seen: set[int] | None = None) -> None:
    """Deallocate every reachable owned TT tensor exactly once."""

    if seen is None:
        seen = set()

    def visit(item: Any) -> None:
        if item is None:
            return
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, ttnn.Tensor):
            ttnn.deallocate(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
        else:
            owned_values = getattr(item, "owned_tensor_values", None)
            if callable(owned_values):
                visit(owned_values())

    visit(value)


def best_effort_deallocate_owned_tensors(
    value: Any,
    completed: set[int] | None = None,
) -> list[BaseException]:
    """Release all reachable TT tensors while retaining successful progress."""

    if completed is None:
        completed = set()
    failures: list[BaseException] = []
    visiting: set[int] = set()

    def visit(item: Any) -> None:
        if item is None:
            return
        item_id = id(item)
        if isinstance(item, ttnn.Tensor):
            if item_id in completed:
                return
            try:
                ttnn.deallocate(item)
            except BaseException as error:
                failures.append(error)
            else:
                completed.add(item_id)
            return
        if item_id in visiting:
            return
        if isinstance(item, dict):
            visiting.add(item_id)
            for nested in item.values():
                visit(nested)
            visiting.remove(item_id)
        elif isinstance(item, (list, tuple, set)):
            visiting.add(item_id)
            for nested in item:
                visit(nested)
            visiting.remove(item_id)
        else:
            owned_values = getattr(item, "owned_tensor_values", None)
            if callable(owned_values):
                visiting.add(item_id)
                visit(owned_values())
                visiting.remove(item_id)

    visit(value)
    return failures


def release_orphans(orphans: list[TensorResourceOrphan]) -> list[BaseException]:
    """Retry orphan releases in place, retaining only incomplete entries."""

    failures: list[BaseException] = []
    remaining: list[TensorResourceOrphan] = []
    for orphan in orphans:
        orphan_failures = best_effort_deallocate_owned_tensors(
            orphan.values,
            orphan.deallocated_tensor_ids,
        )
        failures.extend(orphan_failures)
        if orphan_failures:
            remaining.append(orphan)
    orphans[:] = remaining
    return failures


def attach_cleanup_failures(
    primary: BaseException,
    failures: Iterable[BaseException],
    *,
    note: str = "Cleanup also encountered {count} failure(s)",
) -> None:
    """Attach cleanup failures without replacing the primary exception."""

    failures = tuple(failures)
    if not failures:
        return
    previous = tuple(getattr(primary, "cleanup_failures", ()))
    primary.cleanup_failures = previous + failures
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note.format(count=len(failures)))


def raise_cleanup_failures(failures: Iterable[BaseException]) -> None:
    """Raise the first cleanup failure while retaining every additional one."""

    failures = tuple(failures)
    if not failures:
        raise ValueError("at least one cleanup failure is required")
    primary = failures[0]
    attach_cleanup_failures(
        primary,
        failures[1:],
        note="Cleanup also encountered {count} additional failure(s)",
    )
    raise primary
