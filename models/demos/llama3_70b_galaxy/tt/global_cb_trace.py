# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class _GlobalCBLayout:
    data_address: int
    config_address: int
    size: int
    buffer_type: object
    sender_core_type: str
    mapping: tuple

    @classmethod
    def from_buffer(cls, global_cb):
        mapping = tuple(
            (
                sender.x,
                sender.y,
                tuple(sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in receivers.ranges())),
            )
            for sender, receivers in global_cb.sender_receiver_core_mapping()
        )
        return cls(
            global_cb.buffer_address(),
            global_cb.config_address(),
            global_cb.size(),
            global_cb.buffer_type(),
            global_cb.sender_core_type(),
            mapping,
        )


class GlobalCBTraceState:
    """Own the capture-time GCB reservation until the model's traces are released.

    During prefill, suspension lends its contents to static CBs while ordinary
    allocations still avoid the addresses. Decode resumes the same allocation.
    """

    def __init__(self):
        self.global_cb = None
        self._layout = None
        self._trace_groups = {}

    def validate(self, global_cb):
        if global_cb is None or global_cb.is_suspended():
            raise RuntimeError("Decode trace requires a live global circular buffer; switch to decode mode first")
        layout = _GlobalCBLayout.from_buffer(global_cb)
        if self._layout is not None and layout != self._layout:
            raise RuntimeError(
                "Global circular buffer layout changed after trace capture. "
                f"Expected {self._layout}; got {layout}. "
                "Recreate the model and capture its traces again before decode."
            )
        return layout

    def record_traces(self, group, trace_ids, global_cb):
        trace_ids = tuple(trace_ids)
        if not trace_ids:
            self._trace_groups.pop(group, None)
            return
        layout = self.validate(global_cb)
        if self._layout is None:
            self._layout = layout
        self.global_cb = global_cb
        self._trace_groups[group] = trace_ids

    def restore(self, global_cb):
        self.validate(global_cb)
        # Resumption makes the backing buffers live again. Only traces captured
        # with this exact reservation are exempt; prefill keeps its checks.
        for trace_ids in self._trace_groups.values():
            for trace_id in trace_ids:
                global_cb.acknowledge_restored_trace(trace_id)
