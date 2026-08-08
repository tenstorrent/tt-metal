# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run a DiffusionGemma prefill for its K/V writes only, skipping the lm_head.

DG prefill exists to write the prompt's K/V into the frozen cache. Its logits are
never read: ``generate.prefill_prompt_tokens`` deallocates the forward's return
value on the next line, and ``chunked_prefill`` deallocates every non-final
chunk's. The shared backbone nevertheless ends every non-decode forward in
``Gemma4Model._apply_lm_head`` -- a 262k-vocab matmul, three softcap elementwise
ops, and, at tp > 1, a TP all-gather.

That all-gather is the expensive part, and not because of its bytes (a
[1, 1, 32, vocab/tp] bf16 gather is single-digit MB). ``ccl_allgather``
(``models/demos/gemma4/tt/ccl.py:131``) calls plain ``ttnn.all_gather``, whose
program factory calls ``create_global_semaphore`` three times
(``ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp:37-44``),
and ``GlobalSemaphore::setup_buffer`` is a blocking ``enqueue_write_mesh_buffer``
ending in ``FDMeshCommandQueue::finish_nolock`` -- a full command-queue drain. That
is the same mechanism ``tt/ccl.py`` removed from the ~90 all_reduces per forward
(291a241ae76); this call site survived it because it sits in the SHARED lm_head,
which DG cannot reach by re-pointing imports, and because it is reached on a
program-cache MISS only, so it is intermittent rather than constant.

Measured on tt-shield benchmarks run 30640405931 (2026-07-31, 30 layers, QB2,
32 served requests): prefill is bimodal, with a fast path that scales correctly
with prompt length (0.134 / 0.332 / 0.575 / 1.058 / 2.099 s at cache_len 128 /
1024 / 2048 / 4096 / 8192) and a slow path of 5.8-29.3 s that is prompt-length
INDEPENDENT (median 17.2 s at 2048 > 13.8 s at 4096 > 10.7 s at 8192) -- one
event per prefill, which is exactly this call's cardinality. Traced denoise is
immune for the documented reason: trace replay never rebuilds a program, so the
48 per-block ``_apply_lm_head`` calls inside the denoise trace never reach the
factory, and the denoise step stays flat at 195.8 ms while prefill swings 100x.

The skip uses the backbone's own ``_prefill_trace_mode`` hook, which returns
post-norm hidden states before the last-token slice and the lm_head
(``models/demos/gemma4/tt/model.py:860``). It exists because baking a 262k-vocab
lm_head into a prefill trace is ~40x the model body at 4k tokens; gemma4's own
generator sets it the same way (``gemma4/tt/generator.py:150-163``). Nothing in
the shared tree is edited.

This is output-neutral by construction, not by measurement: K/V is written inside
the decoder layers, before ``self.norm`` and before the branch this flag takes, and
the tensor it changes is one the caller deallocates unread. The DENOISE path is
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
