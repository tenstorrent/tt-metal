# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Debug tracing for the distributed dispatch layer.

Usage:
    from ttml.distributed.debug import DispatchTracer, dispatch_trace

    # Option 1: context manager
    with DispatchTracer() as tracer:
        y = model(x)
    for entry in tracer.entries:
        print(entry)

    # Option 2: global toggle
    dispatch_trace.enable()
    y = model(x)
    dispatch_trace.disable()
    for entry in dispatch_trace.entries:
        print(entry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .layout import Layout

logger = logging.getLogger("ttml.distributed.dispatch")


@dataclass
class TraceEntry:
    """Single dispatch event."""

    op_name: str
    input_layouts: List[Optional[Layout]]
    rule_name: Optional[str]
    plan: Any  # ShardingPlan or None
    redistributions: List[Dict[str, Any]]
    post_collectives: List[Dict[str, Any]]
    output_layout: Optional[Layout]

    def __repr__(self) -> str:
        parts = [f"op={self.op_name}"]
        parts.append(f"inputs={self.input_layouts}")
        if self.rule_name:
            parts.append(f"rule={self.rule_name}")
        if self.redistributions:
            parts.append(f"redist={self.redistributions}")
        if self.post_collectives:
            parts.append(f"post_ccl={self.post_collectives}")
        parts.append(f"output={self.output_layout}")
        return f"TraceEntry({', '.join(parts)})"


class DispatchTrace:
    """In-memory trace buffer for dispatch events."""

    def __init__(self):
        self._enabled: bool = False
        self._entries: List[TraceEntry] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entries(self) -> List[TraceEntry]:
        return list(self._entries)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._entries.clear()

    def record(self, entry: TraceEntry) -> None:
        if self._enabled:
            self._entries.append(entry)
            logger.debug("%s", entry)


# Global singleton
dispatch_trace = DispatchTrace()


class DispatchTracer:
    """Context manager that enables tracing for a scope and collects entries."""

    def __init__(self):
        self._entries_before: int = 0

    def __enter__(self) -> "DispatchTracer":
        self._entries_before = len(dispatch_trace._entries)
        dispatch_trace.enable()
        return self

    def __exit__(self, *exc):
        dispatch_trace.disable()
        return False

    @property
    def entries(self) -> List[TraceEntry]:
        return dispatch_trace._entries[self._entries_before :]
