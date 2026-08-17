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
from .dflash_drafter import DFlashDrafter, DFlashDrafterCache, build_noise_ids


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
            "verify_seconds": self.verify_seconds,
            "total_seconds": self.total_seconds,
            "ms_per_token": self.ms_per_token,
            "tokens_per_second": self.tokens_per_second,
        }


class DFlashRunner:
    """Drives target + drafter for one user on cache slot 0."""

    def __init__(self, generator, drafter: DFlashDrafter) -> None:
        self.generator = generator
        self.model = generator.model
        self.drafter = drafter
        self.config = drafter.config

    # ------------------------------------------------------------------ helpers

    def _tap_layers(self) -> tuple[int, ...]:
        return self.config.target_layer_ids

    def _taps_to_host(self, num_rows: int) -> torch.Tensor:
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
            host = host.reshape(1, -1, self.config.hidden_size)[:, :num_rows, :]
            pieces.append(host.float())
        return torch.cat(pieces, dim=-1)

    def _upload_context(self, context: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            context.reshape(1, 1, *context.shape[-2:]).to(torch.bfloat16),
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
        ids = build_noise_ids(anchor_token_id, self.config.block_size, self.config.mask_token_id)
        tt_ids, _ = self.model.prefill_tokens_to_device(ids)
        embedded = self.model._embed(tt_ids)
        ttnn.deallocate(tt_ids)
        return embedded

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
        hidden = model.prefill_forward(
            embedded, page_table=tt_page_table, user_id=0, start_pos=0, keep_sliding_tails=True
        )
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

        # ------------------------------------------------------------- loop
        while len(produced) < max_new_tokens:
            if stop_on_eos and produced[-1] in eos:
                break
            if anchor_pos + block >= model.config.max_seq_len:
                break

            context_len = int(context_host.shape[1])
            context_positions = torch.arange(context_start, context_start + context_len)
            noise_positions = torch.arange(anchor_pos, anchor_pos + block)

            t0 = time.perf_counter()
            tt_context = self._upload_context(context_host)
            noise = self._noise_embeds(produced[-1])
            drafter_out = self.drafter.forward_cached(
                noise,
                tt_context,
                context_positions=context_positions,
                noise_positions=noise_positions,
                cache=drafter_cache,
            )
            ttnn.deallocate(tt_context)
            candidates = self._candidate_ids(drafter_out)
            ttnn.deallocate(drafter_out)
            stats.draft_seconds += time.perf_counter() - t0

            # ------------------------------------------------------ verify
            t0 = time.perf_counter()
            verify_ids = [produced[-1]] + candidates  # anchor + 15 candidates == 16 positions
            model.arm_hidden_state_taps(self._tap_layers())
            tt_tokens, _ = model.prefill_tokens_to_device(verify_ids)
            embedded = model.embed_prefill(tt_tokens)
            ttnn.deallocate(tt_tokens)
            hidden = model.prefill_forward(
                embedded,
                page_table=tt_page_table,
                user_id=0,
                start_pos=anchor_pos,
                continuation=True,
                keep_sliding_tails=True,
            )
            stats.target_forwards += 1
            rows = model.prefill_all_logits(hidden, prompt_len=len(verify_ids))
            target_argmax = self._argmax_rows(rows[0], len(verify_ids))
            for row in rows:
                ttnn.deallocate(row)
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
            context_host = self._taps_to_host(num)
            ttnn.deallocate(hidden)
            context_start = anchor_pos
            # The new anchor is the last committed token.
            anchor_pos = anchor_pos + result.n_committed

        drafter_cache.release()
        model.release_hidden_state_taps()
        model.release_sliding_tails()
        ttnn.deallocate(tt_page_table)

        produced = produced[:max_new_tokens]
        stats.tokens = len(produced)
        stats.total_seconds = time.perf_counter() - started
        return produced, stats
