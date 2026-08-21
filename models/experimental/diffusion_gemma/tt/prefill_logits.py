# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run a DiffusionGemma prefill for its K/V writes only, skipping the lm_head.

DG prefill exists to write the prompt's K/V into the frozen cache. Its logits are
never read: ``generate.prefill_prompt_tokens`` deallocates the forward's return
value on the next line, and ``chunked_prefill`` deallocates every non-final
chunk's. The shared backbone nevertheless ends every non-decode forward in
``Gemma4Model._apply_lm_head`` -- a 262k-vocab matmul, three softcap elementwise
ops, and, at tp > 1, a TP all-gather.

That all-gather is the expensive part, and not because of its bytes: on a
program-cache MISS its plain ``ttnn.all_gather`` builds GlobalSemaphores whose
setup is a blocking write ending in a full command-queue drain, and it sits in
the SHARED lm_head, which DG cannot reach by re-pointing imports -- so the cost
is intermittent rather than constant.

The skip uses the backbone's own ``_prefill_trace_mode`` hook, which returns
post-norm hidden states before the last-token slice and the lm_head; gemma4's
own generator sets it the same way. Nothing in the shared tree is edited.

This is output-neutral by construction: K/V is written inside the decoder
layers, before ``self.norm`` and before the branch this flag takes, and the
tensor it changes is one the caller deallocates unread. The DENOISE path is
unaffected -- ``denoise_forward`` calls ``_apply_lm_head`` directly rather than
through the forward, so it never consults this flag.
"""

from __future__ import annotations

from contextlib import contextmanager

_FLAG = "_prefill_trace_mode"


@contextmanager
def discard_prefill_logits(model, *, enabled: bool = True):
    """Make ``model``'s non-decode forward return post-norm hidden states.

    Use only around a prefill forward whose return value is discarded: with the
    flag set the forward skips the last-token slice, the lm_head and its TP
    all-gather, so what comes back is [1, 1, seq, hidden] post-norm hidden states
    instead of [1, 1, 32, vocab] logits. Callers that need logits (the final
    chunked-prefill chunk, every denoise step) must pass ``enabled=False`` or not
    use this at all.

    ``enabled=False`` is a no-op passthrough so a caller can keep one ``with``
    statement instead of branching. The previous flag state is restored exactly,
    including the case where the attribute was never set, so this composes with
    gemma4's traced-prefill generator setting the same flag.
    """
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
                # The attribute lives on the class, not the instance; leaving the
                # instance shadow at its pre-existing value is the faithful
                # restore in that case.
                setattr(model, _FLAG, previous)
