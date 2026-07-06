# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched canvas decode for DiffusionGemma (#47557) — model-side, behind ``DG_BATCH_DECODE``.

Today serving runs a **single active canvas** (batch=1): the denoise loop processes one
256-token canvas. #47557 processes **B>1 canvases through the denoise loop simultaneously**.
This module is the model-side prototype (the vLLM multi-request wiring is #47488, out of scope).

The diffusion-delta **decision kernels** (`gumbel_max`/`argmax`, `token_entropy`,
`entropy_budget_accept`, `renoise`) are already batch-generic — they reduce/select over the vocab
or canvas axis, per ``(b, ·)`` row (see the design note). So a `[B, 1, C, vocab]` logits tensor
drives one batched denoise step producing a `[B, C]` committed argmax. What differs by mode is how
the per-step logits are produced from the shared gemma4 backbone:

* ``loop`` (default, robust): the backbone is the shared ``models/demos/gemma4`` graph, whose
  prefill-shaped ops (MoE experts especially) are hard ``[1,1,seq,H]`` (batch=1) and MUST NOT be
  edited. So each canvas row runs the proven single-canvas logits path and the rows are stacked into
  ``[B, 1, C, vocab]``. Per-canvas self-conditioning state is kept per row. This guarantees
  correctness and independence and is the correctness gate.
* ``dim0`` (opt-in): feeds ``[B, 1, C, …]`` straight through the DG-local denoise forward, which
  batches the attention on dim0 and loops only the batch=1 backbone pieces (MoE/LM-head/norms)
  internally. Exercises the generalized DG-local ops; reported but not the gate (its viability
  depends on the shared op batch support).

Either way, the **committed argmax for canvas i is independent of canvas j** (no cross-canvas
leakage) — the device correctness check asserts B=2 == two B=1 runs, bit-exact per row.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt.denoise_loop import _ids_to_torch, run_fixed_denoise_steps


def batched_decode_enabled() -> bool:
    """Whether the opt-in batched canvas decode is on (``DG_BATCH_DECODE``, default OFF)."""
    return os.environ.get("DG_BATCH_DECODE", "0").lower() in ("1", "true", "yes", "on")


def _require_enabled() -> None:
    if not batched_decode_enabled():
        raise RuntimeError("batched canvas decode is opt-in; set DG_BATCH_DECODE=1 to run B>1 canvases (#47557)")


def _slice_canvas_row(canvas_tokens, row: int):
    """Return the ``[1, 1, C, 1]`` slice of a ``[B, 1, C, 1]`` canvas at batch index ``row``."""
    return ttnn.slice(
        canvas_tokens,
        [row, 0, 0, 0],
        [row + 1, canvas_tokens.shape[1], canvas_tokens.shape[2], canvas_tokens.shape[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class BatchedDenoiseLogitsFn:
    """Batched logits callback that loops the single-canvas adapter over B rows.

    Wraps one :class:`~...tt.denoise_forward.DenoiseLogitsAdapter` (built against the shared
    batch=1 prompt prefix) and threads **per-row** self-conditioning state, so row i's logits
    depend only on canvas i. Returns the stacked ``[B, 1, C, vocab]`` logits the batched decision
    kernels consume. The stacked tensor is a fresh copy (``owns_logits`` -> ``False``), so the loop
    frees it each step while the per-row prev-logits stay retained for the next step's self-cond.
    """

    def __init__(self, adapter, batch: int):
        if batch <= 0:
            raise ValueError("batch must be positive")
        self.adapter = adapter
        self.batch = batch
        self._prev: List[Optional[ttnn.Tensor]] = [None] * batch
        # Per-row self-conditioning is threaded via ``self._prev`` + ``adapter.prev_logits``
        # (the non-trace ``__call__`` path). The adapter's TRACE-SAFE self-cond path instead
        # keeps a SINGLE persistent in-place signal buffer (``_signal_read_write_bufs``,
        # ping_pong=False → read buf IS write buf) that ignores ``prev_logits`` — so running B
        # canvases through the shared adapter would let row 0's soft-embedding signal leak into
        # row 1 within the same step (confirmed: identical-input B=2 gave committed[0]!=committed[1]).
        # Force the per-row prev_logits path so each canvas's self-cond depends only on its own
        # previous logits (functionally identical; the batched prototype runs eager, not traced).
        if getattr(adapter, "trace_safe_self_conditioning", False):
            adapter.trace_safe_self_conditioning = False

    @property
    def q_rope_offset(self):
        return self.adapter.q_rope_offset

    @q_rope_offset.setter
    def q_rope_offset(self, value):
        self.adapter.q_rope_offset = value

    def __call__(self, canvas_tokens, step: int):
        row_logits: List[ttnn.Tensor] = []
        for row in range(self.batch):
            canvas_row = _slice_canvas_row(canvas_tokens, row)
            # Restore this canvas's self-cond state, run the proven single-canvas path.
            self.adapter.prev_logits = self._prev[row]
            logits_row = self.adapter(canvas_row, step)  # retains adapter.prev_logits = logits_row
            canvas_row.deallocate(True)
            # Detach the retained per-row logits from the shared adapter so the next row does not
            # free it; keep it as this row's prev-logits for the next step's self-conditioning.
            self._prev[row] = self.adapter.prev_logits
            self.adapter.prev_logits = None
            row_logits.append(logits_row)
        # The returned stacked logits are FREED by the denoise loop each step
        # (``owns_logits`` -> False), so they must not share a device buffer with any
        # retained per-row prev-logits (needed for the next step's ``condition()``;
        # freeing an aliased buffer -> TT_FATAL is_allocated in self_conditioning
        # ``_soft_embedding_chunked``). At B=1 ``ttnn.concat`` of a single tensor ALIASES
        # its input buffer and returns a *distinct Python object* over it, so a
        # ``combined is prev`` identity check misses the alias — clone the lone row into a
        # fresh buffer instead. For B>1 ``concat`` already allocates a fresh output.
        if self.batch == 1:
            combined = ttnn.clone(row_logits[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            combined = ttnn.concat(row_logits, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return combined

    def owns_logits(self, logits) -> bool:
        # The returned stacked logits are a fresh concat, not retained; let the loop free it.
        return False

    def reset(self):
        for row in range(self.batch):
            if self._prev[row] is not None:
                self._prev[row].deallocate(True)
                self._prev[row] = None
        reset = getattr(self.adapter, "reset", None)
        if callable(reset):
            reset()


def run_batched_denoise_block(
    adapter,
    init_canvas,
    config: DiffusionConfig,
    *,
    start_pos: int,
    batch: int,
    noise_tokens_fn=None,
    gumbel_noise_fn=None,
    mode: str = "loop",
) -> torch.Tensor:
    """Run one batched denoise block over ``init_canvas`` ``[B, 1, C, 1]``.

    Runs the trace-safe fixed-step loop (``run_fixed_denoise_steps``) — no early-halt, so the batch
    is per-row independent (early-halt is the only batch-coupling in the eager loop; see the design
    note). Returns the committed clean-argmax tokens ``[B, C]`` on host. Does **not** commit K/V to
    the cache (that needs B caches = #47488); the committed argmax tokens are the model-side output.
    """
    _require_enabled()
    if mode not in ("loop", "dim0"):
        raise ValueError(f"unknown batched decode mode {mode!r}")
    adapter.q_rope_offset = start_pos
    if mode == "loop":
        logits_fn = BatchedDenoiseLogitsFn(adapter, batch)
    else:
        logits_fn = adapter
    committed_dev = run_fixed_denoise_steps(
        logits_fn,
        init_canvas,
        config,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
    )
    committed_host = _ids_to_torch(committed_dev)
    committed_dev.deallocate(True)
    if committed_host.dim() == 1:
        committed_host = committed_host.unsqueeze(0)
    return committed_host
