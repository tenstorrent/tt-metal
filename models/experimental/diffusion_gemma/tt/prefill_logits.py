# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Skip the unused LM head on DiffusionGemma K/V-only prefill forwards.

DG prefill writes prompt K/V into the frozen cache and discards the forward
result. The shared backbone otherwise runs the large LM head and its plain TP
all-gather, which can create global semaphores and block on a command-queue
drain. The backbone's existing ``_prefill_trace_mode`` hook returns post-norm
hidden states before the last-token slice and LM head, after all K/V writes.
"""

from __future__ import annotations

from contextlib import contextmanager

_FLAG = "_prefill_trace_mode"


@contextmanager
def discard_prefill_logits(model, *, enabled: bool = True):
    """Make a K/V-only non-decode forward skip its unused LM head."""
    if not enabled:
        yield
        return

    had_flag = hasattr(model, _FLAG)
    previous = getattr(model, _FLAG, None)
    setattr(model, _FLAG, True)
    try:
        yield
    finally:
        if had_flag:
            setattr(model, _FLAG, previous)
        else:
            try:
                delattr(model, _FLAG)
            except AttributeError:
                # The flag can live on the class. Restore the inherited value
                # rather than leaving an instance shadow pinned to True.
                setattr(model, _FLAG, previous)
