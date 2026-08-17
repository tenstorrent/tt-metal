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
        aligned_verify: bool = False,
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
        #: Restart the verify forward at the page-block boundary below the anchor,
        #: threading sliding K/V tails, instead of re-forwarding the whole prefix from 0.
        #:
        #: **INCOMPLETE - default off.**  This is the change the arithmetic demands: the
        #: from-0 verify is 106.8 ms of a 157.6 ms iteration (68 %) against a break-even
        #: budget of 3.12 tokens x 23.31 ms = 72.7 ms, and it *grows* with the prefix, so
        #: DFlash gets worse the longer it runs.  The scaffold here works end to end and
        #: the two constraints people expect to block it do not: ``_chunk_page_table``
        #: already shifts the page table by ``start_pos / block_size``, and a sliding tail
        #: for an earlier position is a *prefix* of the one already held, which
        #: ``trim_sliding_tails`` already produces.  Two measured problems remain:
        #:
        #: 1. **It is slower, not faster.**  39 of the target's 52 layers are sliding, so
        #:    trimming every iteration costs 78 device slices plus 39 tail concats, and
        #:    verify went 106.8 -> 157 ms/iteration: the tail bookkeeping costs more than
        #:    the shorter forward saves.  The fix is to stop paying it per iteration --
        #:    ``prefill_forward`` *consumes* the tail it is handed, which forces a fresh
        #:    trim every call, so it needs a borrow mode that neither frees the tail nor
        #:    replaces it.  The trim is then only needed when ``aligned_start`` actually
        #:    advances, i.e. once per ``page_block_size`` tokens rather than 20x more often.
        #: 2. **One committed token is wrong**, reproducibly.  At OSL 128 the aligned path
        #:    emits two tokens (34302, 14166) where from-0 and greedy emit one (26382), at
        #:    produced index 32 / absolute position 99, after which the streams re-align.
        #:    Committed tokens are the target's own argmax, so a wrong one means the verify
        #:    forward saw wrong history, not that the drafter guessed badly. Ruled out by
        #:    inspection: the chunked-SDPA offset (q and k chunk sizes are both shrunk
        #:    until they divide ``start_pos``), tile-padding garbage in the tail (always
        #:    trimmed off, since ``aligned_start' <= anchor_pos + block``), and rejected
        #:    candidates inside the tail (everything below ``aligned_start'`` is either a
        #:    prefix token or an accepted candidate, whose K/V is by definition correct).
        #:
        #: Reproduce with ``--verify aligned``; the gate is token equality against
        #: ``--verify from-zero``, which is exact and much sharper than acceptance rate.
        self.aligned_verify = bool(aligned_verify)
        #: Absolute position where the sliding K/V tails currently held by the model end,
        #: or None when none are held.  Tracked here rather than asked of the model
        #: because the tail's *length* is min(window, end) and so does not reveal it.
        self._tail_end: int | None = None

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
            pieces.append(host.float())
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
        return ttnn.from_torch(
            host.to(torch.bfloat16),
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

    def _argmax_rows(self, logits: ttnn.Tensor, rows: int) -> list[int]:
        """Host argmax over the first ``rows`` rows of a vocab-sharded logits tile."""
        gathered = self.model.gather_and_untilize_logits(logits)
        host = self.model.logits_to_torch(gathered, gathered=True)
        ttnn.deallocate(gathered)
        return [int(torch.argmax(host[r]).item()) for r in range(rows)]

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
        logits = self.model.lm_head.forward(padded)
        ttnn.deallocate(padded)
        ids = self._argmax_rows(logits, block)
        ttnn.deallocate(logits)
        return ids[1:]  # position 0 is the anchor; it predicts candidate 0

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

        # ---------------------------------------------------------- prefill
        model.arm_hidden_state_taps(self._tap_layers())
        tt_tokens, _ = model.prefill_tokens_to_device(prompt)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        # Retain the tail: the very first verify already restarts at a page boundary
        # below the prompt length (67 -> 64), so it needs a tail from this call.
        prompt_padded_rows = int(embedded.shape[2])
        hidden = model.prefill_forward(
            embedded,
            page_table=tt_page_table,
            user_id=0,
            start_pos=0,
            keep_sliding_tails=self.aligned_verify,
        )
        self._tail_end = prompt_padded_rows if self.aligned_verify else None
        stats.target_forwards += 1
        # Context for iteration 0 is the whole prompt: positions 0..L-1.
        context_host = self._taps_to_host(prompt_len)
        logits = model.prefill_logits(hidden, last_token_index=prompt_len - 1)
        ttnn.deallocate(hidden)
        anchor = self._argmax_rows(logits, model.row_within_tile(prompt_len - 1) + 1)[
            model.row_within_tile(prompt_len - 1)
        ]
        ttnn.deallocate(logits)

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
                width = context_bucket(valid) if self.pad_context else valid
                tt_context = self._upload_context(accumulated_context, pad_to=width)
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
            stats.draft_forward_seconds += time.perf_counter() - t_forward
            t_candidates = time.perf_counter()
            candidates = self._candidate_ids(drafter_out)
            ttnn.deallocate(drafter_out)
            stats.draft_candidates_seconds += time.perf_counter() - t_candidates
            stats.draft_seconds += time.perf_counter() - t0

            # ------------------------------------------------------ verify
            #
            # ``paged_fill_cache`` always writes from virtual block 0 of the page
            # table it is handed, so a multi-token prefill MUST start on a page-block
            # boundary -- ``start_pos=67`` raises.  Speculation needs a forward at an
            # arbitrary position, so align *down* to the block boundary and re-forward
            # the committed tokens in between.  That is close to free at batch 1: the
            # target is weight-bandwidth bound, so an 80-row forward costs about what a
            # 32-row one does, and re-forwarding committed tokens rewrites byte-identical
            # K/V at the same positions.
            t0 = time.perf_counter()
            # Restart the verify forward at the nearest page-block boundary at or below
            # the anchor, instead of re-forwarding the whole prefix from 0.  Two
            # constraints have to be met at once, and both already have machinery:
            #
            #  * ``paged_fill_cache`` writes from virtual block 0 of the page table it is
            #    handed, so ``start_pos`` must be a multiple of the page block size.
            #    ``FunctionalDecoder._chunk_page_table`` shifts the row by
            #    ``start_pos / block_size`` and enforces exactly that.
            #  * A sliding-window layer refuses ``start_pos > 0`` without the previous
            #    call's K/V tail, because the paged chunked SDPA has no sliding-window
            #    mask and cannot recover the window from the cache.
            #
            # The tail bookkeeping is what made this look hard: ``prefill_forward``
            # emits its tail at the *end* of the call, i.e. at ``aligned_start + padded``,
            # while the next verify restarts *earlier* than that.  But a sliding tail is a
            # contiguous run of K/V rows, so the tail for an earlier position is a
            # **prefix** of the one we hold -- exactly what ``trim_sliding_tails`` already
            # does for tile-padded prompts.  So: keep the tail on every forward, then trim
            # it back to the next restart point.
            #
            # Re-forwards at most ``page_block_size - 1`` committed rows instead of the
            # entire prefix, which also rewrites byte-identical K/V at those positions.
            # Bounded to sequences inside the sliding window: past it the tail holds
            # ``[end - window, end)`` and the rows an earlier position needs were never
            # retained, which ``trim_sliding_tails`` raises on rather than approximating.
            full_sequence = prompt + produced
            if self.aligned_verify:
                page_block = int(model.config.page_block_size)
                aligned_start = (anchor_pos // page_block) * page_block
            else:
                aligned_start = 0
            lead = anchor_pos - aligned_start
            verify_ids = full_sequence[aligned_start:anchor_pos] + [produced[-1]] + candidates
            assert len(verify_ids) == lead + block, (len(verify_ids), lead, block)

            # Hand the layers a tail ending exactly at ``aligned_start``;
            # ``sliding_kv_tail_len(start_pos)`` is ``min(window, start_pos)`` and the
            # shape check is exact, so an untrimmed tail is rejected outright.
            continuation = aligned_start > 0
            if continuation:
                if self._tail_end is None or self._tail_end < aligned_start:
                    raise AssertionError(
                        f"verify needs sliding tails ending at {aligned_start} but holds "
                        f"{self._tail_end}; every forward must run with keep_sliding_tails=True"
                    )
                model.trim_sliding_tails(aligned_start, self._tail_end)
                self._tail_end = aligned_start
            else:
                model.release_sliding_tails()
                self._tail_end = None

            model.arm_hidden_state_taps(self._tap_layers())
            tt_tokens, _ = model.prefill_tokens_to_device(verify_ids)
            embedded = model.embed_prefill(tt_tokens)
            ttnn.deallocate(tt_tokens)
            # ``prefill_tokens_to_device`` tile-pads, so the forward writes K/V for the
            # padded length and the tail it retains ends there, not at len(verify_ids).
            padded_rows = int(embedded.shape[2])
            hidden = model.prefill_forward(
                embedded,
                page_table=tt_page_table,
                user_id=0,
                start_pos=aligned_start,
                continuation=continuation,
                keep_sliding_tails=self.aligned_verify,
            )
            if self.aligned_verify:
                self._tail_end = aligned_start + padded_rows
            stats.target_forwards += 1
            rows = model.prefill_all_logits(hidden, prompt_len=len(verify_ids))
            all_argmax: list[int] = []
            for tile_index, row in enumerate(rows):
                remaining = len(verify_ids) - tile_index * TILE_ROWS
                all_argmax.extend(self._argmax_rows(row, min(TILE_ROWS, remaining)))
                ttnn.deallocate(row)
            # Rows [lead, lead + block) are the anchor and the 15 candidates.
            target_argmax = all_argmax[lead : lead + block]
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
            context_host = self._taps_to_host(num, offset=lead)
            # Positions anchor_pos..anchor_pos+n_matches follow the rows already
            # accumulated, so a plain append keeps row i at absolute position i --
            # the invariant forward_padded relies on.
            accumulated_context = torch.cat([accumulated_context, context_host], dim=1)
            ttnn.deallocate(hidden)
            context_start = anchor_pos
            # The new anchor is the last committed token.
            anchor_pos = anchor_pos + result.n_committed

        drafter_cache.release()
        # Disarm, not merely drain: release_hidden_state_taps() frees captured tensors but
        # leaves _tap_indices set, so a later non-speculative decode would keep capturing
        # and hit the sharded-clone failure above.
        model.arm_hidden_state_taps(None)
        model.release_sliding_tails()
        ttnn.deallocate(tt_page_table)

        produced = produced[:max_new_tokens]
        stats.tokens = len(produced)
        stats.total_seconds = time.perf_counter() - started
        return produced, stats
