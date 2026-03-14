# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared memory-tracking helpers."""

import ttml

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


# =====================================================================
# Custom autograd: MemorySnapshotFunction (identity with memory tracking)
# =====================================================================


class MemorySnapshotFunction(ttml.autograd.Function):
    """Identity op that captures a MemoryUsageTracker snapshot on forward
    and/or backward.  Zero computational overhead — just passes through the
    tensor value and gradient unchanged.
    """

    @staticmethod
    def forward(ctx, x, fwd_label, bwd_label):
        ctx.bwd_label = bwd_label
        if fwd_label:
            MemoryUsageTracker.snapshot(fwd_label)
        return x.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.bwd_label:
            MemoryUsageTracker.snapshot(ctx.bwd_label)
        return grad_output


def memory_snapshot(x, fwd_label="", bwd_label=""):
    """Identity wrapper that records memory snapshots during forward/backward.

    Inserts a no-op node into the autograd graph.  When *fwd_label* is set a
    snapshot is taken during the forward pass; when *bwd_label* is set a
    snapshot is taken when the gradient flows back through this point.
    """
    if not fwd_label and not bwd_label:
        return x
    return MemorySnapshotFunction.apply(x, fwd_label, bwd_label)


def print_memory_report(title: str = "Memory Usage Report"):
    """End capture, print the report, and clear state."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    MemoryUsageTracker.print_memory_usage()
    MemoryUsageTracker.clear()
    print("=" * 70)
    print()


def finalize_memory(
    memory_guard, label: str = "COMPLETE", title: str = "Memory Usage Report"
):
    """End capture, print report, release guard.  No-op if *memory_guard* is ``None``."""
    if memory_guard is None:
        return
    MemoryUsageTracker.end_capture(label)
    print_memory_report(title)
    memory_guard.release()
