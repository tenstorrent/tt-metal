# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware memory tracker snapshots."""

from __future__ import annotations

import os

import ttml


_MEMORY_TRACKING_ENV = "TTML_TRACK_MEMORY"
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker
_snapshot_sequence = 0


def memory_tracking_enabled() -> bool:
    return os.environ.get(_MEMORY_TRACKING_ENV, "0") == "1"


def _unique_snapshot_label(label: str) -> str:
    global _snapshot_sequence
    _snapshot_sequence += 1
    return f"{label}__{_snapshot_sequence:06d}"


class MemorySnapshotFunction(ttml.autograd.Function):
    """Identity op that snapshots memory usage on forward and backward."""

    @staticmethod
    def forward(ctx, x, fwd_label, bwd_label):
        ctx.bwd_label = bwd_label
        if fwd_label:
            MemoryUsageTracker.snapshot(_unique_snapshot_label(fwd_label))
        return x.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.bwd_label:
            MemoryUsageTracker.snapshot(_unique_snapshot_label(ctx.bwd_label))
        return grad_output


def memory_snapshot(x, fwd_label: str = "", bwd_label: str = ""):
    """Pass ``x`` through while recording memory snapshots when enabled."""
    if not memory_tracking_enabled() or (not fwd_label and not bwd_label):
        return x
    return MemorySnapshotFunction.apply(x, fwd_label, bwd_label)
