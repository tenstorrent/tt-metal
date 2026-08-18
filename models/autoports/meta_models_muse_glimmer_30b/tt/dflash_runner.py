# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DFlash speculative decoding on device.

Composes the pieces that were validated independently: the TTNN drafter
(:mod:`dflash_drafter`, device PCC 13/13), the target's hidden-state taps
(:meth:`MuseGlimmerModel.arm_hidden_state_taps`), and the accept rule
(:mod:`dflash_accept`, 71 unit tests).

The per-iteration bookkeeping below is not invented; it was read off a trace of
the reference ``DFlashTokenCandidateGenerator`` driving the real models, because
the index arithmetic is subtle enough that a plausible-looking guess would cost
acceptance rate rather than fail::

    iter  ctx_len  ctx_positions  noise_positions  cache_before  committed
    0          67          0..66           67..82             0          3
    1           3         67..69           70..85            67         15
    2          15         70..84           85..100           70         11

The invariants that produces:

* **Iteration 0** passes the *whole prompt's* hidden states as context
  (positions ``0..L-1``), and the first target-sampled token is the anchor at
  position ``L``.  The noise window is ``L .. L+15``.
* **Every later iteration** passes only the *newly accepted* hidden states, taken
  as ``verify_hidden[:num]`` with ``num == n_matches + 1``.  Those are the
  **anchor plus the accepted candidates** - positions ``P .. P+n_matches`` - not
  the positions of the committed tokens themselves.  The drafter's cache
  accumulates them, so the context argument shrinks to just the delta.
* The new anchor is the **last committed token**, at position
  ``P + n_matches + 1``.

One target forward per iteration verifies 16 positions (anchor + 15 candidates)
and yields 16 argmax values, which is exactly the ``len(candidates) + 1`` contract
:func:`accept_block` requires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

import ttnn

from .dflash_accept import accept_block
from .dflash_drafter import DFlashDrafter, DFlashDrafterCache, build_noise_ids, context_bucket

#: The LM head and the tile-padded prefill path both work in 32-row M tiles.
TILE_ROWS = 32


@dataclass
class DFlashStats:
    """Per-generation speculation accounting."""

    tokens: int = 0
    iterations: int = 0
    target_forwards: int = 0
    matches: list[int] = field(default_factory=list)
    committed: list[int] = field(default_factory=list)
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    total_seconds: float = 0.0
    #: ``draft_seconds`` split up.  Worth separating because the drafter *forward*
    #: turned out to be a minority of it: the candidate step runs the target's LM head
    #: over a 202k vocabulary and pulls the result to host for an argmax, which is not
    #: obviously part of "drafting" but is charged to it.
    draft_noise_seconds: float = 0.0
    draft_forward_seconds: float = 0.0
    draft_candidates_seconds: float = 0.0
    #: ``verify_seconds`` split the same way, because "verify" is not all target
    #: forward: it also runs the LM head over every verified row and pulls a
    #: 202k-wide logits tile to host per 32-row tile just to take an argmax.
    verify_forward_seconds: float = 0.0
    verify_logits_seconds: float = 0.0
    #: Pulling the five tapped hidden-state tensors to host and concatenating them.
    #: Charged to neither draft nor verify, so it hid in the unaccounted remainder.
    taps_seconds: float = 0.0
    #: Uploading the (bucket-padded) accumulated context back to device.
    context_upload_seconds: float = 0.0

    @property
    def accepted_per_target_forward(self) -> float:
        return self.tokens / self.target_forwards if self.target_forwards else 0.0

    @property
    def mean_matches(self) -> float:
        return sum(self.matches) / len(self.matches) if self.matches else 0.0

    @property
    def ms_per_token(self) -> float:
        return 1000.0 * self.total_seconds / self.tokens if self.tokens else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.total_seconds if self.total_seconds else 0.0

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "iterations": self.iterations,
            "target_forwards": self.target_forwards,
            "accepted_per_target_forward": self.accepted_per_target_forward,
            "mean_matches": self.mean_matches,
            "matches": self.matches,
            "committed": self.committed,
            "draft_seconds": self.draft_seconds,
            "draft_noise_seconds": self.draft_noise_seconds,
            "draft_forward_seconds": self.draft_forward_seconds,
            "draft_candidates_seconds": self.draft_candidates_seconds,
            "verify_seconds": self.verify_seconds,
            "verify_forward_seconds": self.verify_forward_seconds,
            "verify_logits_seconds": self.verify_logits_seconds,
            "taps_seconds": self.taps_seconds,
            "context_upload_seconds": self.context_upload_seconds,
            "total_seconds": self.total_seconds,
            "ms_per_token": self.ms_per_token,
            "tokens_per_second": self.tokens_per_second,
        }


class DFlashRunner:
    """Drives target + drafter for one user on cache slot 0."""

    #: Cap on the verify forward's width; see the comment at the verify forward.
    DEFAULT_MAX_VERIFY_ROWS = 2048

    def __init__(
        self,
        generator,
        drafter: DFlashDrafter,
        *,
        max_verify_rows: int | None = None,
        padded_drafting: bool = True,
        pad_context: bool = True,
        aligned_verify: bool = True,
        uncapped_argmax: bool = True,
        verify_mode: str = "prefill",
        trace_verify: bool = True,
        verify_width: int = 256,
        verify_rows: int = 32,
        max_verify_traces: int = 24,
    ) -> None:
        self.generator = generator
        self.model = generator.model
        self.drafter = drafter
        self.config = drafter.config
        self.max_verify_rows = int(max_verify_rows or self.DEFAULT_MAX_VERIFY_ROWS)
        #: Drive the drafter at a bucketed, constant context width instead of feeding it
        #: the per-iteration delta against a growing K/V cache.
        #:
        #: Every distinct shape a ttnn op sees costs a program compilation, and the
        #: incremental path produces a **new cache length every iteration**, so it
        #: recompiles for as long as generation continues.  Bucketing bounds the whole
        #: generation to seven shapes.  Measured over 128 tokens / 41 blocks, ISL 67:
        #:
        #: | path        | ms/drafter call | accepted/forward | mean matches | t/s/u |
        #: |-------------|-----------------|------------------|--------------|-------|
        #: | incremental | 671             | 3.05             | 2.10         | 3.93  |
        #: | padded      | 120             | 3.05             | 2.10         | 12.91 |
        #:
        #: **Acceptance is identical**, so bucketing is free accuracy-wise: the wider
        #: attention reduction does perturb the drafter at the 1e-3 level, but it does
        #: not cost acceptance.  A 48-token run suggested otherwise (4.00 vs 2.82) and
        #: was wrong -- over 11 blocks acceptance on one prompt spans 2.8-4.0 across
        #: mathematically equivalent configurations, so it takes ~40 blocks before the
        #: metric discriminates anything.  ``padded-exact`` remains as the control.
        #:
        #: The incremental path's cost is also easy to under-measure: it timed 44.5 ms
        #: per call over 48 tokens purely because earlier runs had left ttnn's **on-disk**
        #: kernel cache warm for that exact shape sequence. Extending to 128 tokens
        #: introduces unseen shapes and it returns to 671 ms. Any drafter measurement
        #: must therefore state its generation length and cache state.
        self.padded_drafting = bool(padded_drafting)
        #: Whether the accumulated-context path also pads to a bucket.
        #:
        #: Setting this False keeps the whole-prefix path but sizes the tensor to the
        #: exact context length, which is the control that separates two very
        #: different explanations for the acceptance drop padding causes: a bug in the
        #: accumulation/positions (would persist without padding) versus padding
        #: changing the attention reduction width and flipping near-tied argmaxes
        #: (would vanish without padding).
        self.pad_context = bool(pad_context)
        #: Take every argmax from the **pre-softcap** logits.
        #:
        #: The LM head computes ``T * tanh(x)``, which is strictly monotonic and so
        #: cannot change an argmax in exact arithmetic -- the reason the first
        #: implementation reused the full head for candidates.  In bf16 it can and does.
        #: Spacing near 1.0 is ``2**-8`` = 0.0039 while ``tanh(x) > 0.996`` for
        #: ``x > 3.1``, so every strong logit saturates to the *same* bf16 value and the
        #: winner becomes whichever index the reduction reaches first.
        #:
        #: Speculation is the workload that cannot afford it: a candidate is accepted only
        #: when the drafter's argmax equals the target's, so ties on both sides make
        #: agreement a coin flip.  HF's candidate path uses the raw ``nn.Linear``, and the
        #: CPU oracle accepts 4.41 tokens per target forward on the same prompt and length
        #: where this port accepted 2.67 -- a 40 % deficit that no amount of drafter
        #: fidelity explained (F8).
        #:
        #: Applies to the *verify* argmax as well as to candidate selection, since a tie
        #: on the target's side costs acceptance exactly as much as one on the drafter's.
        #: Sampling is untouched: the cap shapes the distribution and only its effect on
        #: argmax is a no-op.
        self.uncapped_argmax = bool(uncapped_argmax)
        #: ``"prefill"`` (default) verifies the block with a prefill forward.
        #: ``"decode"`` / ``"decode_eager"`` verify it as one batched decode step and are
        #: **kept only as a recorded negative result** -- they are not sound.
        #:
        #: The idea was strong and the arithmetic still is.  A prefill forward costs 55-67 ms
        #: measured flat from 32 to 256 rows and start positions 0 to 1024, i.e. it is
        #: host-dispatch bound, while a *traced* decode step through the same 52 layers
        #: costs 23.3 ms; and the port documents that ``DECODE_ROWS`` "is not the batch size
        #: and is deliberately independent of it", which measurement confirmed (70.35 ms at
        #: 16 active rows against 72.12 ms at 1).  So putting the anchor and its 15
        #: candidates in 16 decode rows sharing one page-table row, with
        #: ``current_pos[u] = anchor_pos + u``, should verify the whole block for the price
        #: of one decode step -- a 64.5 -> 24 ms change, which is the entire break-even gap.
        #:
        #: It does not hold: **decode rows do not reliably observe each other's same-step
        #: K/V writes.**  Row ``u`` needs the K/V that rows ``< u`` write in the same op,
        #: and two runs of the identical step, each from a freshly re-seeded cache,
        #: disagreed at row 14 -- and the winner differed from the prefill reference by a
        #: **0.59** logit gap, far outside the bf16 near-tie floor.  End to end it produced
        #: degenerate output (token 1574 repeated) whose acceptance *looked* excellent
        #: (blocks of 15/15) precisely because repetition is trivial to draft.
        #:
        #: Two traps worth keeping, because both made it look correct:
        #: running the prefill reference *before* the decode step pre-warms the cache for
        #: exactly those positions, so the decode never has to chain and scores 0/16; and
        #: running the decode twice without re-seeding does the same to the second run.
        #: ``tests/dflash_decode_verify_probe.py`` re-seeds before every measurement.
        self.verify_mode = str(verify_mode)
        #: Capture the verify forward once as a trace and replay it, instead of issuing it
        #: eagerly every iteration.  **Default off: it does not work in this loop.**
        #:
        #: The reasoning is sound and the shape trick is real.  A prefill forward costs
        #: 55-67 ms *flat* from 32 to 256 rows because it is host-dispatch bound, and flat
        #: cost means a **fixed-width** verify is free -- so padding every from-zero verify
        #: to ``verify_width`` with ``start_pos = 0`` makes the whole thing one static
        #: graph, capturable as a single trace.  Tracing is the only mechanism that removes
        #: host issue here, and the port measured a warmed prefill replay at 44.96 ms
        #: against 59.80 ms eager.
        #:
        #: What stops it is the rest of the loop.  A live trace requires that nothing else
        #: allocate device buffers -- ttnn warns "Allocating device buffers is unsafe due to
        #: the existence of an active trace. These buffers may be corrupted once a trace is
        #: executed" -- and DFlash allocates on every iteration: the drafter's activations,
        #: the context upload, the noise embeddings, the logits tiles.  With the trace live
        #: the run wedged rather than returning.  The shipped decode trace coexists with its
        #: loop only because that loop allocates *nothing*: four persistent inputs, one
        #: persistent logits output.
        #:
        #: So this is worth roughly 15 ms/iteration and needs the DFlash loop converted to
        #: persistent buffers throughout first -- drafter activations included -- which is a
        #: larger change than the trace itself.
        self.trace_verify = bool(trace_verify)
        #: Padded row count of the traced verify.  The whole sequence must fit, which the
        #: ``max_verify_rows`` guard enforces; making it larger costs nothing on device but
        #: does retain a wider hidden/tap set.
        self.verify_width = int(verify_width)
        #: Rows in the traced verify window.  **32, and it must stay 32**: the
        #: DRAM-sharded decode matmul asserts ``M == 1`` (exactly one tile row), so a
        #: traced 32-row prefill costs 24.48 ms -- one decode step -- while 64 rows falls
        #: back to mcast2d and costs 40.99.  The candidate count bends around this, not
        #: the window.
        self.verify_rows = int(verify_rows)
        #: How many 32-row windows may hold a trace at once.  Beyond this the verify falls
        #: back to the eager path rather than evicting: releasing a trace mid-generation
        #: while its addresses are still live is what wedged the board twice here.
        self.max_verify_traces = int(max_verify_traces)
        #: Fixed drafter context width for the whole generation, or 0 to grow through
        #: CONTEXT_BUCKETS.  Set when the verify is traced: a bucket growth is a
        #: bucket-sized allocation between replays, inside the trace's address range.
        #: Fixed drafter context width for the whole generation, or 0 to grow through
        #: CONTEXT_BUCKETS.  Set from the run length when the verify is traced.
        self.pinned_context_bucket = 0
        self._prefill_trace: dict | None = None
        #: ``aligned_start -> {id, hidden, taps}`` for the 32-row traced verify.
        #:
        #: One trace per distinct start position, because ``start_pos`` is baked into the
        #: graph -- prefill RoPE slices its tables with host indices, and the chunked SDPA
        #: offset is a dispatch-time constant unless it is fed the device-tensor form.
        #: That is affordable: the window advances 32 positions at a time, so a 128-token
        #: generation needs about five, each ~140 ms to capture and reused for every
        #: iteration that lands in the same 32-row block.
        self._verify_traces: dict[int, dict] = {}
        #: The traced verify's two inputs.  Both addresses are **baked into the trace**
        #: at capture, so they must be allocated once, outlive the trace, and never be
        #: rebound.  The first implementation allocated them per ``generate()`` and freed
        #: the page table at the end of the call, which left every later replay reading a
        #: buffer the allocator had already handed to something else -- the same ordering
        #: that produced the fabric ERISC assert documented in ``generator.py``, and the
        #: reason the first traced run wedged the board rather than returning a wrong
        #: number.
        self._verify_tokens: ttnn.Tensor | None = None
        self._verify_page_table: ttnn.Tensor | None = None
        self._owns_verify_page_table = False
        #: DRAM bytes/bank just after capture; the invariant every replay must restore.
        self._alloc_baseline: int | None = None
        #: Label of the most recent allocation checkpoint, so the assertion names the stage.
        self._alloc_where = "?"
        self._alloc_reported = 0
        #: Times the per-shard argmax had to fall back because a padding column won.
        self._argmax_pad_fallbacks = 0
        #: Bytes/bank of post-capture allocation tolerated across a replay.  Not a
        #: licence: a *growing* drift is a leak and must fail, while a small flat one is
        #: a lazily-created cache that the replay is unlikely to collide with.  Set to 0
        #: to make any drift fatal.
        self.alloc_drift_budget = 256 * 1024
        #: Lazily captured trace for the verify decode step, with taps armed.  The shipped
        #: decode trace cannot be reused: it is captured *without* hidden-state taps, and
        #: DFlash needs them to build the drafter's context.
        self._verify_trace: dict | None = None
        #: Restart the verify forward at the page-block boundary below the anchor,
        #: threading sliding K/V tails, instead of re-forwarding the whole prefix from 0.
        #:
        #: This is the change the arithmetic demands: the from-0 verify is 106.8 ms of a
        #: 157.6 ms iteration (68 %) against a break-even budget of 3.12 tokens x
        #: 23.31 ms = 72.7 ms, and it *grows* with the prefix, so DFlash gets worse the
        #: longer it runs.  The aligned restart re-forwards at most
        #: ``page_block_size - 1`` committed rows regardless of position.
        #:
        #: Neither of the two constraints the first implementation recorded as blocking
        #: this is real.  ``_chunk_page_table`` already shifts the page table by
        #: ``start_pos / block_size``.  And no sliding K/V tail is needed at all: while
        #: the sequence sits inside the 2048 window, every sliding layer's window
        #: excludes nothing, so it *is* a full-attention layer over the chunk and the
        #: decoder routes it through the paged path
        #: (``FunctionalDecoder.sliding_window_is_inert``).  An earlier attempt did
        #: thread tails, via ``trim_sliding_tails``, and was both slower -- 78 device
        #: slices per iteration across the 39 sliding layers took verify to 157 ms -- and
        #: wrong by one committed token.  Removing the tails removed both problems.
        #:
        #: Past the window this must fall back to real tail threading; the
        #: ``max_verify_rows`` guard refuses rather than silently attending a truncated
        #: history.  The gate is token equality with ``--verify from-zero``, which is
        #: exact and far sharper than acceptance rate: the earlier tail bug moved
        #: acceptance by only 0.2 while moving tokens outright.
        self.aligned_verify = bool(aligned_verify)

    # ------------------------------------------------------------------ helpers

    def _tap_layers(self) -> tuple[int, ...]:
        return self.config.target_layer_ids

    def _taps_to_host(self, num_rows: int, *, offset: int = 0) -> torch.Tensor:
        """Concatenate the tapped hidden states on the last dim: ``[1, num_rows, 5*H]``.

        Assembled on host.  The tap tensors are tile-padded to 32 rows while
        ``num_rows`` is an arbitrary accepted count, and an unaligned device slice
        of a TILE tensor is a gather anyway.  At 16 rows this is ~1 MB per
        iteration; the prompt-sized first iteration is the one worth moving on
        device later.
        """
        taps = self.model.take_hidden_state_taps()
        pieces = []
        for layer_idx in self._tap_layers():
            tensor = taps[layer_idx]
            host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=0))[0:1]
            ttnn.deallocate(tensor)
            host = host.reshape(1, -1, self.config.hidden_size)[:, offset : offset + num_rows, :]
            # Kept in bf16 rather than widened to float32: it is uploaded as bf16, and at
            # bucket 256 a float32 accumulator costs an 8.5M-element convert *per
            # iteration* on the host for nothing.
            pieces.append(host.to(torch.bfloat16))
        return torch.cat(pieces, dim=-1)

    def _upload_context(self, context: torch.Tensor, *, pad_to: int | None = None) -> ttnn.Tensor:
        """Upload context hidden states, optionally zero-padded to a fixed width.

        Padding is what keeps every drafter op at a constant shape across
        iterations; the pad rows are removed by the drafter's mask, never by a
        slice, because a slice would reintroduce a varying shape.
        """
        host = context.reshape(1, 1, *context.shape[-2:])
        if pad_to is not None:
            rows = int(host.shape[2])
            if rows > pad_to:
                raise ValueError(f"context has {rows} rows, more than the {pad_to}-row bucket")
            if rows < pad_to:
                padding = torch.zeros(1, 1, pad_to - rows, int(host.shape[3]), dtype=host.dtype)
                host = torch.cat([host, padding], dim=2)
        if host.dtype != torch.bfloat16:
            host = host.to(torch.bfloat16)
        return ttnn.from_torch(
            host,
            device=self.model.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh_device),
        )

    def _noise_embeds(self, anchor_token_id: int) -> ttnn.Tensor:
        """Raw embedding lookup for ``[anchor] + [mask] * 15``.

        ``model._embed`` is the bare gather.  ``embed_prefill`` would additionally
        apply the embedding norm, which HF deliberately bypasses for the drafter
        ("the assistant needs embedding without norm").
        """
        block = self.config.block_size
        ids = build_noise_ids(anchor_token_id, block, self.config.mask_token_id)
        tt_ids, _ = self.model.prefill_tokens_to_device(ids)
        embedded = self.model._embed(tt_ids)
        ttnn.deallocate(tt_ids)
        # prefill_tokens_to_device pads to a 32-row tile.  The window MUST be exactly
        # block_size wide: attention here is bidirectional, so pad rows would be
        # attended to as keys by the real queries and corrupt them, rather than being
        # harmlessly ignored the way they are on the target's causal prefill path.
        rows = int(embedded.shape[2])
        if rows == block:
            return embedded
        trimmed = ttnn.slice(embedded, [0, 0, 0, 0], [1, 1, block, int(embedded.shape[3])])
        ttnn.deallocate(embedded)
        return trimmed

    def _taps_from(self, taps: dict, num_rows: int, *, offset: int = 0) -> torch.Tensor:
        """Assemble the drafter context from already-held tap tensors, without freeing them.

        ``_taps_to_host`` drains and deallocates, which is right for the prefill path where
        every forward produces fresh taps.  The traced decode path must not: its tap tensors
        are **persistent trace outputs**, rewritten in place by each replay, so freeing them
        would invalidate the trace.
        """
        pieces = []
        for layer_idx in self._tap_layers():
            host = ttnn.to_torch(taps[layer_idx], mesh_composer=ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=0))[
                0:1
            ]
            host = host.reshape(1, -1, self.config.hidden_size)[:, offset : offset + num_rows, :]
            pieces.append(host.to(torch.bfloat16))
        return torch.cat(pieces, dim=-1)

    def _ensure_verify_inputs(self, tt_page_table: ttnn.Tensor) -> None:
        """Allocate the traced verify's persistent inputs exactly once.

        Both are read by the captured graph at addresses fixed at capture time, so
        reallocating either between generations silently detaches the trace from its
        inputs -- the token buffer would be refreshed while the trace reads the previous
        one, and freeing the page table leaves the replay reading whatever the allocator
        handed out next.  The latter is a use-after-free inside a paged attention op and
        wedges the board rather than failing cleanly.
        """
        if self._verify_tokens is not None:
            return
        model = self.model
        # Sized to the traced window, not to verify_width: every trace reads this one
        # buffer, and the shipped traced path is the 32-row window.
        host_ids = torch.full((1, self.verify_rows), model.embed_pad_id, dtype=torch.int32)
        self._verify_tokens = ttnn.from_torch(
            host_ids,
            device=model.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
        )
        # The page table is the caller's; the trace now depends on it, so generate()
        # must not free it while the trace lives.
        self._verify_page_table = tt_page_table
        self._owns_verify_page_table = True

    def release_prefill_verify_trace(self) -> None:
        """Release the verify trace and the inputs whose addresses it baked in.

        Order matters and is the whole point: the trace must go first.  Freeing a buffer
        a live trace still references is what produced the board wedge, and this project
        already has the same finding recorded for a cloned KV cache.
        """
        if self._prefill_trace is None:
            return
        try:
            ttnn.release_trace(self.model.mesh_device, self._prefill_trace["id"])
            self.model.note_trace_released()
        finally:
            self._prefill_trace = None
            if self._verify_tokens is not None:
                ttnn.deallocate(self._verify_tokens)
            self._verify_tokens = None
            self._verify_page_table = None
            self._owns_verify_page_table = False

    def _alloc_note(self, where: str) -> None:
        """Record a checkpoint label and, when tracing, the DRAM delta since capture."""
        self._alloc_where = where
        if self._alloc_baseline is not None:
            delta = self._dram_allocated() - self._alloc_baseline
            if False:
                from loguru import logger as _logger

                _logger.info(f"alloc delta after {where}: {delta} bytes/bank")

    def _dram_allocated(self) -> int:
        """Bytes allocated per DRAM bank, for the allocation-invariant assertion.

        The contract a live trace imposes is that anything allocated between replays is
        **dead before the next replay** -- its intermediates are freed at capture but
        their addresses stay baked into the graph, so a buffer handed out from that range
        while a replay is in flight is overwritten by the replay.  Comparing this value
        immediately before each replay against its value just after capture turns that
        from a code review into a runtime check, and names any leak in bytes.
        """
        view = ttnn.get_memory_view(self.model.mesh_device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_allocated_per_bank)

    def _ensure_prefill_verify_trace(self) -> None:
        """Capture the fixed-width, from-zero verify forward once, with taps armed.

        Warm-compile first (it executes, so it must not be the capture), then capture
        ``embed_prefill`` + ``prefill_forward`` reading the persistent token buffer.  The
        hidden state and the five taps are the graph's own outputs, so their addresses are
        baked in and every replay refreshes them in place -- nothing here may free them.
        """
        if self._prefill_trace is not None:
            return
        model = self.model
        tap_layers = self._tap_layers()
        tokens = self._verify_tokens
        page_table = self._verify_page_table
        # Make this the only live trace.  How much *address range* sits under the
        # allocation rule is what decided the shipped prefill-trace configuration in
        # generator.py; one graph is the smallest that range gets, and it also stops the
        # verify's persistent inputs from landing inside another trace's replay footprint.
        self.generator._release_decode_trace()
        self.generator._release_prefill_traces()

        model.arm_hidden_state_taps(tap_layers)
        model.release_sliding_tails()
        warm_embedded = model.embed_prefill(tokens)
        warm = model.prefill_forward(warm_embedded, page_table=page_table, user_id=0, start_pos=0)
        ttnn.deallocate(warm)
        for tensor in model.take_hidden_state_taps().values():
            ttnn.deallocate(tensor)
        ttnn.synchronize_device(model.mesh_device)

        model.release_sliding_tails()
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=0)
        model.arm_hidden_state_taps(tap_layers)
        embedded = model.embed_prefill(tokens)
        hidden = model.prefill_forward(embedded, page_table=page_table, user_id=0, start_pos=0)
        taps = model.take_hidden_state_taps()
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(model.mesh_device)
        model.note_trace_captured()
        self._prefill_trace = {"id": trace_id, "hidden": hidden, "taps": taps}
        self._alloc_baseline = self._dram_allocated()

    def _argmax_range(self, hidden: ttnn.Tensor, start_row: int, count: int) -> list[int]:
        """Argmax for ``[start_row, start_row + count)`` of a prefill hidden state.

        Only the one or two 32-row tiles that actually cover the block are projected, so
        the 202k-wide head runs twice at most however wide the traced forward is.
        """
        model = self.model
        out: list[int] = []
        first_tile = start_row // TILE_ROWS
        last_tile = (start_row + count - 1) // TILE_ROWS
        for tile in range(first_tile, last_tile + 1):
            row = model._slice_rows(hidden, tile * TILE_ROWS, aligned=True)
            normed = model.final_norm.forward(row)
            ttnn.deallocate(row)
            logits = model.lm_head.forward(normed, apply_softcap=not self.uncapped_argmax)
            ttnn.deallocate(normed)
            ids = self._argmax_rows(logits, TILE_ROWS)
            ttnn.deallocate(logits)
            for index, value in enumerate(ids):
                absolute = tile * TILE_ROWS + index
                if start_row <= absolute < start_row + count:
                    out.append(value)
        return out

    def _trace_for(self, aligned_start: int) -> dict | None:
        """Capture (once) and return the verify trace for this 32-row window.

        Returns ``None`` when the cache is full, so the caller falls back to the eager
        verify rather than evicting -- releasing a trace mid-generation while its
        addresses are still baked into buffers is the hazard that wedged the board twice
        in this project.
        """
        entry = self._verify_traces.get(aligned_start)
        if entry is not None:
            return entry
        # Keep exactly ONE live trace.  The window advances monotonically -- once the
        # anchor leaves a 32-row block it never returns -- so an older trace is dead
        # weight, and what matters is how much *address range* sits under the allocation
        # rule: every live trace's footprint is memory the loop must not be handed
        # between replays.  Holding several took a 48-token run that worked to a
        # 128-token run that hung, at three live traces and at five alike.  Released
        # before the next capture, never during a replay.
        self._release_all_verify_traces()
        model = self.model
        tap_layers = self._tap_layers()
        rows = self.verify_rows

        # Warm-compile before EVERY capture, not just the first.  Skipping it for later
        # windows looks safe -- same shapes, only start_pos differs -- and is fatal:
        # "TT_FATAL: Cannot load new binaries during trace".  The chunked SDPA offset is a
        # dispatch-time constant, so a new start_pos is a NEW program, and capturing it
        # uncompiled tries to compile inside the capture.  This warm pass is therefore
        # ~66 ms per 32-token window, and it is exactly what a runtime-offset design
        # (chunk_start_idx_tensor plus on-device prefill RoPE) would remove, along with
        # the capture itself.
        #
        # Deleting the *body* of this and keeping the comment is not a hypothetical: it is
        # what produced the ``Cannot load new binaries during trace capture`` that stalled
        # this path for a day.  The warm pass has to run, not merely be described.
        self.generator._release_decode_trace()
        self.generator._release_prefill_traces()
        model.arm_hidden_state_taps(tap_layers)
        model.release_sliding_tails()
        warm_embedded = model.embed_prefill(self._verify_tokens)
        ttnn.deallocate(
            model.prefill_forward(warm_embedded, page_table=self._verify_page_table, user_id=0, start_pos=aligned_start)
        )
        for tensor in model.take_hidden_state_taps().values():
            ttnn.deallocate(tensor)
        ttnn.synchronize_device(model.mesh_device)

        model.release_sliding_tails()
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=0)
        model.arm_hidden_state_taps(tap_layers)
        embedded = model.embed_prefill(self._verify_tokens)
        hidden = model.prefill_forward(embedded, page_table=self._verify_page_table, user_id=0, start_pos=aligned_start)
        taps = model.take_hidden_state_taps()
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(model.mesh_device)
        model.note_trace_captured()
        entry = {"id": trace_id, "hidden": hidden, "taps": taps, "rows": rows}
        self._verify_traces[aligned_start] = entry
        return entry

    def _release_all_verify_traces(self) -> None:
        """Release the live verify traces, keeping the shared input buffers alive.

        Drained on both sides, which is not optional and is not decoration: the mesh only
        marks allocations safe again once the *last* trace is gone, and a release that has
        not drained leaves the region still claimed.  The next thing this path does is
        warm-compile the following window, and loading binaries while any trace is live
        fails outright with ``TT_FATAL: Cannot load new binaries during trace``.  The
        generator's own trace releases drain the same way for the same reason.
        """
        if not self._verify_traces:
            return
        ttnn.synchronize_device(self.model.mesh_device)
        for entry in self._verify_traces.values():
            try:
                ttnn.release_trace(self.model.mesh_device, entry["id"])
                self.model.note_trace_released()
            except Exception:  # noqa: BLE001 - a failed release must not mask the caller
                pass
        self._verify_traces = {}
        ttnn.synchronize_device(self.model.mesh_device)

    def release_verify_traces(self) -> None:
        """Release every 32-row verify trace, then the buffers they baked in."""
        self._release_all_verify_traces()
        if self._verify_tokens is not None:
            ttnn.deallocate(self._verify_tokens)
        self._verify_tokens = None
        self._verify_page_table = None
        self._owns_verify_page_table = False

    def _verify_traced32(self, full_sequence, candidates, anchor_pos: int, stats):
        """Verify the block as a **32-row** traced prefill at a page-aligned start.

        This is the shape that pays.  Measured warm, the same graph traced:

            rows   eager ms   traced ms
              32      66.77      24.48     <- one decode step (23.3 ms)
              64      61.16      40.99
             128      60.65      47.28

        32 rows is a cliff rather than a slope: the DRAM-sharded decode matmul asserts
        ``M == 1`` -- exactly one tile row -- so at 32 rows the prefill already dispatches
        the *decode* projections and the *decode* collectives, while 64 rows falls back to
        mcast2d at roughly half the DRAM bandwidth.  Anything wider than one tile throws
        most of the win away, which is why the window is fixed at ``verify_rows`` and the
        candidate count bends instead.

        The window is ``[aligned_start, aligned_start + 32)`` with
        ``aligned_start = 32 * floor(anchor_pos / 32)``, so it holds ``lead`` already
        committed rows, the anchor, and ``31 - lead`` candidate slots.  That averages 15.5
        usable candidates against the drafter's 15, i.e. today's block for free, and it is
        self-correcting: a cramped iteration (``lead`` near 31) commits few tokens, which
        moves the anchor into the next block and hands the following iteration a full one.

        Re-forwarding the ``lead`` committed rows is free -- the cost is flat in rows -- and
        rewrites byte-identical K/V at those positions.
        """
        model = self.model
        rows = self.verify_rows
        page_block = int(model.config.page_block_size)
        aligned_start = (anchor_pos // page_block) * page_block
        lead = anchor_pos - aligned_start
        usable = max(0, rows - lead - 1)
        n_cand = min(len(candidates), usable)
        used = list(candidates[:n_cand])

        entry = self._trace_for(aligned_start)
        if entry is None:
            return None  # cache full: caller falls back to the eager verify

        t_forward = time.perf_counter()
        host_ids = torch.full((1, rows), model.embed_pad_id, dtype=torch.int32)
        ids = list(full_sequence[aligned_start:anchor_pos]) + [full_sequence[anchor_pos]] + used
        host_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        host = ttnn.from_torch(
            host_ids,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self._verify_tokens)
        ttnn.execute_trace(model.mesh_device, entry["id"], cq_id=0, blocking=True)
        stats.verify_forward_seconds += time.perf_counter() - t_forward
        stats.target_forwards += 1

        t_logits = time.perf_counter()
        # One 32-row hidden means one tile, so this is a single LM-head call at a constant
        # shape -- unlike the 256-row form, which walked a different tile offset each
        # iteration.
        target_argmax = self._argmax_range(entry["hidden"], lead, n_cand + 1)
        stats.verify_logits_seconds += time.perf_counter() - t_logits
        return target_argmax, entry["taps"], lead, used

    def _verify_prefill_traced(self, full_sequence, candidates, anchor_pos: int, stats):
        """Replay the fixed-width verify trace, then read the block's rows.

        ``full_sequence[:anchor_pos] + [anchor] + candidates`` is padded to
        ``verify_width``; row index therefore *is* absolute position, which is what makes
        the tap offset and the logits range plain slices.
        """
        model = self.model
        block = self.config.block_size
        ids = list(full_sequence[:anchor_pos]) + [full_sequence[anchor_pos]] + list(candidates)
        if len(ids) > self.verify_width:
            raise NotImplementedError(
                f"traced verify is fixed at {self.verify_width} rows and this sequence needs "
                f"{len(ids)}; raise verify_width (device cost is flat in rows) or fall back to "
                "the eager verify"
            )

        # Captured on first use, after the drafter and the LM head have each run once.
        # Both allocate lazily (CCL semaphores, sharded matmul configs), and anything
        # allocated *after* capture stays live across every replay at an address inside
        # the trace's footprint -- measured at 219 KB/bank when captured any earlier.
        self._ensure_prefill_verify_trace()

        t_forward = time.perf_counter()
        host_ids = torch.full((1, self.verify_width), model.embed_pad_id, dtype=torch.int32)
        host_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        host = ttnn.from_torch(
            host_ids,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self._verify_tokens)
        del host
        self._alloc_note("A: after stage tokens")
        if self._alloc_baseline is not None:
            live = self._dram_allocated()
            drift = live - self._alloc_baseline
            if drift > self.alloc_drift_budget:
                raise AssertionError(
                    f"[{self._alloc_where}] device DRAM grew by {drift} bytes/bank since the verify "
                    f"trace was captured, over the {self.alloc_drift_budget} budget. Something "
                    "allocated between replays and is still live; the replay will overwrite it."
                )
            if drift and drift != self._alloc_reported:
                from loguru import logger as _logger

                _logger.warning(
                    f"verify trace: {drift} bytes/bank live across the replay (budget "
                    f"{self.alloc_drift_budget}). Under the allocation rule that is a buffer the "
                    "replay may overwrite; it is tolerated only because it is flat, not growing."
                )
                self._alloc_reported = drift
        # Blocking: the replay must complete before anything allocated after it can be
        # handed an address inside the trace's footprint.
        ttnn.execute_trace(model.mesh_device, self._prefill_trace["id"], cq_id=0, blocking=True)
        stats.verify_forward_seconds += time.perf_counter() - t_forward
        stats.target_forwards += 1
        self._alloc_note("B: after replay")

        t_logits = time.perf_counter()
        target_argmax = self._argmax_range(self._prefill_trace["hidden"], anchor_pos, block)
        self._alloc_note("C: after argmax_range")
        stats.verify_logits_seconds += time.perf_counter() - t_logits
        self._alloc_note("verify replay + argmax")
        return target_argmax, self._prefill_trace["taps"]

    def _ensure_verify_trace(self) -> None:
        """Capture the verify decode step once, with taps armed.

        Mirrors ``MuseGlimmerGenerator._capture_decode_trace``: warm-compile first (which
        *executes*, so anything it mutated is restaged by the caller afterwards), then
        capture.  ``advance_positions=False`` because the verify restages absolute
        positions every iteration and must not have them incremented underneath it.

        The tensors kept here -- the logits and the five taps -- are the graph's own
        outputs, so their addresses are baked into the trace and every replay refreshes
        them in place.  That is the same contract the decode trace's ``_trace_logits``
        relies on, and it is why nothing in the loop may deallocate them.
        """
        if self._verify_trace is not None:
            return
        model = self.model
        inputs = self.generator._device_inputs
        tap_layers = self._tap_layers()

        model.arm_hidden_state_taps(tap_layers)
        warm = model.ttnn_decode_forward(
            inputs["tokens"],
            inputs["current_pos"],
            inputs["rope_pos_ids"],
            inputs["page_table"],
            advance_positions=False,
        )
        ttnn.deallocate(warm)
        for tensor in model.take_hidden_state_taps().values():
            ttnn.deallocate(tensor)
        ttnn.synchronize_device(model.mesh_device)

        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=0)
        model.arm_hidden_state_taps(tap_layers)
        logits = model.ttnn_decode_forward(
            inputs["tokens"],
            inputs["current_pos"],
            inputs["rope_pos_ids"],
            inputs["page_table"],
            advance_positions=False,
        )
        taps = model.take_hidden_state_taps()
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(model.mesh_device)
        model.note_trace_captured()
        self._verify_trace = {"id": trace_id, "logits": logits, "taps": taps}

    def release_verify_trace(self) -> None:
        if self._verify_trace is None:
            return
        try:
            ttnn.release_trace(self.model.mesh_device, self._verify_trace["id"])
        finally:
            self._verify_trace = None

    def _argmax_rows_sharded(self, logits: ttnn.Tensor, rows: int) -> list[int]:
        """Per-shard argmax. **Measured SLOWER than the gathered path; not used.**

        Correct -- it produced byte-identical tokens over 128 tokens -- and a clear loss:
        26.67 -> 21.12 t/s/u, with the candidate step going 9.7 -> 10.5 ms and the verify
        logits 4.9 -> 6.5 ms.  The premise was that the all-gather of a 32 x 202752 tile
        (~13 MB) dominated; it does not.  Replacing one readback of one small tensor with
        **eight** small per-shard readbacks costs more than the gather saves, which says
        the cost here is per-transfer latency, not bytes.  Folding ids and maxima into a
        single tensor per device would halve the count to four and is the only version
        worth retrying.

        Kept because the reasoning is sound and the measurement is the useful part.
        Reduced **per shard, in place**: nothing crosses the mesh, and what crosses PCIe
        is ``4 x rows`` ids plus ``4 x rows`` maxima rather than a ``rows x 202752`` tile.
        DFlash calls this on every drafted block *and* on the verify block, so it sits on
        the critical path twice per iteration -- 9.6 ms and 5.0 ms respectively.

        The previous version called ``gather_and_untilize_logits`` first, an all-gather of
        a ``32 x 50688`` shard into a ``32 x 202752`` replica (~13 MB landed per device)
        plus an untilize and a 13 MB slice.  Every byte of that is discarded one op later:
        an argmax needs one winner per row, and the winner of a partitioned max is the
        best of the per-partition winners.

        The layout that makes this legal is the one the sampler's device-offset table
        already encodes: shard ``d`` owns the contiguous global range
        ``[d * local_vocab, (d + 1) * local_vocab)`` in mesh device order -- the order
        ``ttnn.get_device_tensors`` returns and ``logits_to_torch`` concatenates in.  So
        ``global_id = d * local_vocab + local_id``.

        THE PADDING.  ``padded_vocab_size`` (202752) exceeds ``vocab_size`` (202048), and
        the 704 extra columns are not tokens.  They all sit in the **last** shard: shards
        0-2 are wholly real at 50688 columns, shard 3 is real for 49984.  A ttnn op
        applies the same slice to every device, so a per-shard reduction cannot simply
        trim the way the gathered path did.  This *detects* a padded winner by index
        instead, which needs no assumption about the padded values at all, and redoes
        only that row from the guarded shard.  It should never fire: those columns are a
        bias-free matmul against exactly-zero weights, so their logits are exactly +0.0,
        which can only win if every one of the 202048 real logits is negative.
        ``_argmax_pad_fallbacks`` counts it -- non-zero after a run is a finding.

        Ties resolve to the lowest global id: shards are scanned in ascending order and
        only a strictly greater max displaces the incumbent.
        """
        model = self.model
        config = model.config
        local = int(config.local_vocab_size)
        vocab = int(config.vocab_size)
        shards = len(ttnn.get_device_tensors(logits))

        width = int(logits.shape[-1])
        if width != local:
            raise ValueError(
                f"_argmax_rows wants vocab-sharded logits ({local} columns per device), got {width}. "
                "A gathered tensor reaching here would have its shard-0 ids read as global ids."
            )
        # Real columns per shard; everything at or above this on that shard is padding.
        valid = [max(0, min(vocab - d * local, local)) for d in range(shards)]

        # max off the TILE tensor (multi-core, and the width is a whole number of tiles so
        # no intra-tile padding joins the reduction); argmax off a ROW_MAJOR copy, because
        # ttnn.argmax on TILE input with dim=rank-1 runs SINGLE-core, which over 50688
        # columns would cost more than the gather this replaces.
        maxima = ttnn.max(logits, dim=-1, keepdim=True)
        untilized = ttnn.untilize(logits, use_multicore=True)
        ids = ttnn.argmax(untilized, dim=-1)
        ttnn.deallocate(untilized)

        shard_ids = [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(ids)]
        shard_max = [ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(maxima)]
        ttnn.deallocate(ids)
        ttnn.deallocate(maxima)
        model.counters["readbacks"] += 1

        out: list[int] = []
        padded_rows: list[int] = []
        for r in range(rows):
            best_d, best_v = -1, None
            for d in range(shards):
                if valid[d] == 0:
                    continue  # a wholly-padded shard owns no tokens and never votes
                value = float(shard_max[d][r])
                if best_v is None or value > best_v:
                    best_v, best_d = value, d
            local_id = int(shard_ids[best_d][r])
            if local_id >= valid[best_d]:
                out.append(-1)
                padded_rows.append(r)
            else:
                out.append(best_d * local + local_id)

        if padded_rows:
            self._argmax_pad_fallbacks += len(padded_rows)
            guarded = {
                d: ttnn.to_torch(ttnn.get_device_tensors(logits)[d]).reshape(-1, local).float()
                for d in range(shards)
                if 0 < valid[d] < local
            }
            model.counters["readbacks"] += 1
            for r in padded_rows:
                best_d, best_v, best_i = -1, None, -1
                for d in range(shards):
                    if valid[d] == 0:
                        continue
                    if d in guarded:
                        value_t, index_t = guarded[d][r, : valid[d]].max(dim=-1)
                        value, index = float(value_t), int(index_t)
                    else:
                        value, index = float(shard_max[d][r]), int(shard_ids[d][r])
                    if best_v is None or value > best_v:
                        best_v, best_d, best_i = value, d, index
                out[r] = best_d * local + best_i
        return out

    def _argmax_rows(self, logits: ttnn.Tensor, rows: int) -> list[int]:
        """Argmax over the first ``rows`` rows of a vocab-sharded logits tile.

        Gathers, trims the vocab padding, and reduces on device, so what crosses PCIe is
        ``rows`` token ids rather than a ``rows x 202752`` float tile.  The trim is not
        optional: ``gather_and_untilize_logits`` returns ``padded_vocab_size`` columns and
        a device argmax over the padded width can return an id that is not a token.

        A per-shard variant that avoids the all-gather entirely is implemented above and
        is **slower** -- see ``_argmax_rows_sharded`` for the measurement.
        """
        model = self.model
        gathered = model.gather_and_untilize_logits(logits)
        vocab = int(model.config.vocab_size)
        shape = tuple(gathered.shape)
        if int(shape[-1]) != vocab:
            trimmed = ttnn.slice(gathered, [0, 0, 0, 0], [shape[0], shape[1], shape[2], vocab])
            ttnn.deallocate(gathered)
            gathered = trimmed
        ids = ttnn.argmax(gathered, dim=-1)
        ttnn.deallocate(gathered)
        host = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1)
        ttnn.deallocate(ids)
        model.counters["readbacks"] += 1
        return [int(host[r]) for r in range(rows)]

    def _candidate_ids(self, drafter_hidden: ttnn.Tensor) -> list[int]:
        """Target ``lm_head`` on the drafter's output, dropping the anchor position.

        HF applies the target's raw ``lm_head``; this port's LM head also folds in
        ``output_multiplier`` and the tanh softcap.  Both are strictly monotonic,
        so **greedy argmax is unchanged** - which is all the candidates are used
        for.  It would matter for sampling, where the distribution differs.
        """
        block = self.config.block_size
        rows = int(drafter_hidden.shape[2])
        # The LM head's matmul contract is one 32-row M tile.
        if rows < TILE_ROWS:
            padded = ttnn.pad(drafter_hidden, [(0, 0), (0, 0), (0, TILE_ROWS - rows), (0, 0)], value=0.0)
        else:
            padded = self.model._slice_rows(drafter_hidden, 0, aligned=True)
        logits = self.model.lm_head.forward(padded, apply_softcap=not self.uncapped_argmax)
        ttnn.deallocate(padded)
        ids = self._argmax_rows(logits, block)
        ttnn.deallocate(logits)
        return ids[1:]  # position 0 is the anchor; it predicts candidate 0

    def _verify_decode(self, anchor_token: int, candidates: list[int], anchor_pos: int, stats):
        """Verify the block with one replay of the traced decode step.

        Row ``u`` carries the token at ``anchor_pos + u`` and is limited to
        ``[0, anchor_pos + u]``, so it attends the candidates before it and not those
        after -- the causal structure of a 16-row prefill, checked position by position
        in ``tests/dflash_decode_verify_probe.py`` (0 mismatches of 16).
        """
        model = self.model
        block = self.config.block_size
        verify_tokens = [int(anchor_token)] + list(candidates)

        t_forward = time.perf_counter()
        self.generator._stage(
            tokens=verify_tokens,
            positions=torch.arange(anchor_pos, anchor_pos + len(verify_tokens)),
        )
        inputs = self.generator._device_inputs
        if self.verify_mode == "decode_eager":
            # Same graph, no trace: isolates a trace-replay fault from a logic fault.
            model.arm_hidden_state_taps(self._tap_layers())
            logits = model.ttnn_decode_forward(
                inputs["tokens"],
                inputs["current_pos"],
                inputs["rope_pos_ids"],
                inputs["page_table"],
                advance_positions=False,
            )
            taps = model.take_hidden_state_taps()
            owned = True
        else:
            ttnn.execute_trace(model.mesh_device, self._verify_trace["id"], cq_id=0, blocking=True)
            logits, taps, owned = self._verify_trace["logits"], self._verify_trace["taps"], False
        stats.verify_forward_seconds += time.perf_counter() - t_forward
        stats.target_forwards += 1

        t_logits = time.perf_counter()
        target_argmax = self._argmax_rows(logits, block)
        stats.verify_logits_seconds += time.perf_counter() - t_logits
        if owned:
            ttnn.deallocate(logits)
            self._eager_taps_owned = taps
        return target_argmax, taps

    def _verify_prefill(self, full_sequence, candidates: list[int], anchor_pos: int, tt_page_table, stats):
        """Verify the block with a prefill forward, restarting at a page-block boundary.

        Kept as the reference path the decode verify is graded against.  It re-forwards up
        to ``page_block_size - 1`` committed rows, and a prefill forward costs 55-67 ms
        whatever its width, so it is ~2.7x the decode step for the same result.
        """
        model = self.model
        block = self.config.block_size
        if self.aligned_verify:
            page_block = int(model.config.page_block_size)
            aligned_start = (anchor_pos // page_block) * page_block
        else:
            aligned_start = 0
        lead = anchor_pos - aligned_start
        verify_ids = list(full_sequence[aligned_start:anchor_pos]) + [full_sequence[anchor_pos]] + list(candidates)
        assert len(verify_ids) == lead + block, (len(verify_ids), lead, block)

        model.arm_hidden_state_taps(self._tap_layers())
        tt_tokens, _ = model.prefill_tokens_to_device(verify_ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        # No sliding K/V tails are threaded and none are needed: inside the 2048 window a
        # sliding layer's window excludes nothing, so the decoder routes it down the paged
        # path exactly as it does a full layer (FunctionalDecoder.sliding_window_is_inert).
        model.release_sliding_tails()
        t_forward = time.perf_counter()
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=aligned_start)
        ttnn.synchronize_device(model.mesh_device)
        stats.verify_forward_seconds += time.perf_counter() - t_forward
        stats.target_forwards += 1

        t_logits = time.perf_counter()
        rows = model.prefill_all_logits(hidden, prompt_len=len(verify_ids), apply_softcap=not self.uncapped_argmax)
        all_argmax: list[int] = []
        for tile_index, row in enumerate(rows):
            remaining = len(verify_ids) - tile_index * TILE_ROWS
            all_argmax.extend(self._argmax_rows(row, min(TILE_ROWS, remaining)))
            ttnn.deallocate(row)
        stats.verify_logits_seconds += time.perf_counter() - t_logits
        return all_argmax[lead : lead + block], hidden, lead

    # -------------------------------------------------------------------- driver

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        *,
        page_table=None,
        stop_on_eos: bool = True,
    ) -> tuple[list[int], DFlashStats]:
        model = self.model
        block = self.config.block_size
        prompt = list(int(t) for t in prompt_token_ids)
        prompt_len = len(prompt)
        eos = tuple(self.generator._eos_ids) if stop_on_eos else ()

        stats = DFlashStats()
        started = time.perf_counter()

        self.generator._invalidate_traces_if_cache_moved()
        self.generator._allocate_device_inputs()
        table = self.generator._coerce_page_table(page_table)
        self.generator._stage(page_table=table)
        slot_row = model.page_table_row(table, 0)
        tt_page_table = model.page_table_row_to_device(slot_row)
        if self.verify_mode.startswith("decode"):
            # Every verify row addresses the SAME sequence: they write different positions
            # inside slot 0, which is what makes a batched decode step a block verify.
            if int(model.config.max_batch_size) < block:
                raise ValueError(
                    f"decode verify needs max_batch_size >= block_size ({block}), got "
                    f"{model.config.max_batch_size}; build the generator with max_batch_size=32"
                )
            shared_rows = slot_row.repeat(int(model.config.max_batch_size), 1)
            self.generator._stage(tokens=[0], positions=torch.zeros(1), page_table=shared_rows)
            if self.verify_mode == "decode":
                self._ensure_verify_trace()

        # ---------------------------------------------------------- prefill
        model.arm_hidden_state_taps(self._tap_layers())
        tt_tokens, _ = model.prefill_tokens_to_device(prompt)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=0)
        stats.target_forwards += 1
        self._alloc_note("after prompt prefill")
        # Context for iteration 0 is the whole prompt: positions 0..L-1.
        context_host = self._taps_to_host(prompt_len)
        logits = model.prefill_logits(hidden, last_token_index=prompt_len - 1, apply_softcap=not self.uncapped_argmax)
        ttnn.deallocate(hidden)
        anchor = self._argmax_rows(logits, model.row_within_tile(prompt_len - 1) + 1)[
            model.row_within_tile(prompt_len - 1)
        ]
        ttnn.deallocate(logits)

        if self.trace_verify and not self.verify_mode.startswith("decode"):
            # Capture only once the model is warm.  The prompt prefill lazily allocates
            # per-length device caches (RoPE tables, norm/memory-config buffers) that
            # would otherwise be created *after* capture and sit live across every
            # replay -- measured at 334 KB/bank, which the allocation assertion catches.
            # Capturing here puts them in the baseline instead.
            #
            # The capture executes the graph, so it overwrites the KV cache for rows
            # 0..verify_width with pad tokens.  That is safe precisely because this verify
            # is from-zero: the first replay re-forwards the whole prefix and rewrites
            # every row it will read.  The prompt's taps and anchor have already been
            # read out above, so nothing else depends on that KV.
            # Fix the drafter context width for the whole generation at the smallest
            # bucket the run can need, so it never grows mid-run.  A growth is a
            # bucket-sized allocation between replays inside the live trace's address
            # range, and it is what hung every earlier 128-token attempt while 48-token
            # runs -- which never leave their first bucket -- worked.
            self.pinned_context_bucket = context_bucket(prompt_len + max_new_tokens)
            self._ensure_verify_inputs(tt_page_table)

        produced = [anchor]
        # Absolute position of the anchor, i.e. of noise slot 0.
        anchor_pos = prompt_len
        # Absolute position of the first context row we are about to pass.
        context_start = 0
        drafter_cache = DFlashDrafterCache(self.config.num_hidden_layers)
        # The padded path is stateless on device, so the accumulated context lives
        # here instead of in a device K/V cache.  ``context_host`` stays the
        # per-iteration delta the incremental path wants; this is the running
        # concatenation of those deltas, i.e. the whole accepted prefix, whose rows
        # are at absolute positions 0..n-1 by construction.
        accumulated_context = context_host

        # ------------------------------------------------------------- loop
        while len(produced) < max_new_tokens:
            if stop_on_eos and produced[-1] in eos:
                break
            if anchor_pos + block >= model.config.max_seq_len:
                break
            if anchor_pos + block > self.max_verify_rows:
                # Two different reasons to stop here, depending on the path.
                #
                # from-0 verify: the forward is O(prefix) and would quietly become the
                # dominant cost.
                #
                # aligned verify: the forward is bounded, but the *tail* is not. Past the
                # sliding window a retained tail holds [end - window, end), so trimming it
                # back to an earlier restart point needs rows that were never retained,
                # and trim_sliding_tails raises rather than approximating. Lifting this
                # means reconstructing the tail from the paged cache instead of carrying
                # it forward.
                detail = (
                    "the retained sliding tail cannot be trimmed back past the window"
                    if self.aligned_verify
                    else "the verify forward re-forwards the whole prefix"
                )
                raise NotImplementedError(
                    f"DFlash verify is bounded to the sliding window: {detail}; position "
                    f"{anchor_pos + block} exceeds max_verify_rows={self.max_verify_rows}. "
                    "Reconstruct the sliding tail from the paged cache to go further."
                )

            context_len = int(context_host.shape[1])
            context_positions = torch.arange(context_start, context_start + context_len)
            noise_positions = torch.arange(anchor_pos, anchor_pos + block)

            t0 = time.perf_counter()
            t_noise = time.perf_counter()
            noise = self._noise_embeds(produced[-1])
            self._alloc_note("noise")
            stats.draft_noise_seconds += time.perf_counter() - t_noise
            t_forward = time.perf_counter()
            if self.padded_drafting:
                # The accumulated prefix, padded up to a bucket so the shape is constant.
                valid = int(accumulated_context.shape[1])
                # Row i must be at absolute position i, so the prefix must end exactly
                # where the noise window begins.  Passing the whole prefix uncached is
                # equivalent to the incremental cache because encoder.fc, output_norm_enc
                # and k_proj are all per-row, so a row's K/V does not depend on how many
                # rows accompanied it -- that linearity is what the cache exploits too.
                if valid != anchor_pos:
                    raise AssertionError(
                        f"accumulated context has {valid} rows but the anchor is at {anchor_pos}; "
                        "context rows and absolute positions have drifted apart"
                    )
                # Held at one width for the whole generation rather than grown through the
                # buckets.  Growing it reallocates a bucket-sized tensor mid-run -- 17 MB
                # at 256 -- and while a verify trace is live that is a large allocation
                # landing in the trace's address range between replays.  It is also the
                # one thing that differs structurally between a 48-token run (bucket stays
                # at 128, works) and a 128-token one (grows to 256, hangs).
                if self.pinned_context_bucket and self.pad_context:
                    width = self.pinned_context_bucket
                else:
                    width = context_bucket(valid) if self.pad_context else valid
                t_up = time.perf_counter()
                tt_context = self._upload_context(accumulated_context, pad_to=width)
                stats.context_upload_seconds += time.perf_counter() - t_up
                drafter_out = self.drafter.forward_padded(
                    noise,
                    tt_context,
                    context_valid=valid,
                    noise_start=anchor_pos,
                )
            else:
                tt_context = self._upload_context(context_host)
                drafter_out = self.drafter.forward_cached(
                    noise,
                    tt_context,
                    context_positions=context_positions,
                    noise_positions=noise_positions,
                    cache=drafter_cache,
                )
            ttnn.deallocate(tt_context)
            # _DFlashLayer.forward rebinds its local name rather than freeing the
            # hidden state it was handed, so `noise` is still live here -- and would stay
            # live across the verify replay, at an address inside the trace's footprint.
            ttnn.deallocate(noise)
            self._alloc_note("after drafter forward")
            stats.draft_forward_seconds += time.perf_counter() - t_forward
            t_candidates = time.perf_counter()
            candidates = self._candidate_ids(drafter_out)
            ttnn.deallocate(drafter_out)
            stats.draft_candidates_seconds += time.perf_counter() - t_candidates
            self._alloc_note("after candidates")
            stats.draft_seconds += time.perf_counter() - t0

            # ------------------------------------------------------ verify
            t0 = time.perf_counter()
            traced32 = None
            if self.trace_verify and not self.verify_mode.startswith("decode"):
                traced32 = self._verify_traced32(prompt + produced, candidates, anchor_pos, stats)
            if traced32 is not None:
                target_argmax, verify_taps, lead, candidates = traced32
                hidden = None
            elif self.verify_mode.startswith("decode"):
                # One traced decode step over the block: the anchor and its candidates at
                # absolute positions anchor_pos..anchor_pos+block-1, all sharing slot 0's
                # page-table row.  Nothing is re-forwarded -- the prefix is already in the
                # paged cache -- so this is O(block) by construction rather than by
                # alignment, and costs one decode step whatever the block size.
                target_argmax, verify_taps = self._verify_decode(produced[-1], candidates, anchor_pos, stats)
                lead, hidden = 0, None
            else:
                target_argmax, hidden, lead = self._verify_prefill(
                    prompt + produced, candidates, anchor_pos, tt_page_table, stats
                )
                verify_taps = None
            stats.verify_seconds += time.perf_counter() - t0

            result = accept_block(
                candidates,
                target_argmax,
                eos_token_ids=eos,
                max_new_tokens=max_new_tokens - len(produced),
            )
            produced.extend(result.tokens)
            stats.iterations += 1
            stats.matches.append(result.n_matches)
            stats.committed.append(result.n_committed)

            # Context for the next iteration: verify_hidden[:n_matches + 1], i.e. the
            # anchor plus the accepted candidates, at positions anchor_pos..+n_matches.
            num = result.n_matches + 1
            t_taps = time.perf_counter()
            if verify_taps is not None:
                # Trace outputs: read, never drain.
                context_host = self._taps_from(verify_taps, num, offset=lead)
            else:
                context_host = self._taps_to_host(num, offset=lead)
            stats.taps_seconds += time.perf_counter() - t_taps
            self._alloc_note("taps")
            # Positions anchor_pos..anchor_pos+n_matches follow the rows already
            # accumulated, so a plain append keeps row i at absolute position i --
            # the invariant forward_padded relies on.
            accumulated_context = torch.cat([accumulated_context, context_host], dim=1)
            if hidden is not None:
                ttnn.deallocate(hidden)
            context_start = anchor_pos
            # The new anchor is the last committed token.
            anchor_pos = anchor_pos + result.n_committed

        drafter_cache.release()
        self.release_verify_traces()
        # Disarm, not merely drain: release_hidden_state_taps() frees captured tensors but
        # leaves _tap_indices set, so a later non-speculative decode would keep capturing
        # and hit the sharded-clone failure above.
        model.arm_hidden_state_taps(None)
        model.release_sliding_tails()
        if not self._owns_verify_page_table:
            ttnn.deallocate(tt_page_table)

        produced = produced[:max_new_tokens]
        stats.tokens = len(produced)
        stats.total_seconds = time.perf_counter() - started
        return produced, stats
