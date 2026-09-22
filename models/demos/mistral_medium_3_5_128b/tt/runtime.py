# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2: the chunked-prefill runtime — ``compile``, ``make_chunk_input``, ``prefill_chunk``.

:class:`MistralModel` already prefills one chunk at an arbitrary ``cached_len``. What this module
adds is the **contract around the chunk schedule**: the model takes ``cached_len`` on faith, and a
caller that hands it the wrong one gets a plausible-looking answer. The block-cyclic KV layout is
addressed from that offset (``tt/attention/kv_cache.py``), so an off-by-one-chunk write lands
somewhere real and silently wrong — a PCC in the nineties on the layers that happened to line up.
Every assertion here exists to turn one of those into a failed call.

The shape of the interface is the prefill engine's runtime contract
(``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` §2), so the serving stage that
follows this bring-up has the piece it calls per chunk already written and measured. Only the three
methods the recipe's P2 asks for are here. The rest of that contract belongs to work this bring-up
does not do: ``capture_trace`` is perf, ``build_kv_chunk_table`` / ``kv_migration_*`` are KV
migration, ``set_layer_completion_sink`` is the pipelined runner — all out of scope (recipe §1),
and a stub that was never driven would be worse than their absence.

**Who owns what.** The runtime does not own the KV cache: it is passed in on every call that
touches it, exactly as the engine does it, which is also what lets one runtime serve several user
slots. The runtime *does* own the chunk bookkeeping, because that is the thing being asserted.

**The contract on ``[actual_start, actual_end)``** — the absolute KV positions of the chunk's real
tokens:

* ``actual_start`` is the cache write offset and must be a multiple of ``chunk_size``. The cache's
  per-chip write is contiguous only at chunk boundaries (``write_kv_chunk``), and the SP split of
  the input is a plain contiguous chunk of the sequence, which equals the block-cyclic layout only
  at chunk-aligned offsets.
* ``actual_end - actual_start`` may be less than ``chunk_size``: the last chunk's tail is pad. The
  padded positions *are* written to the cache — the write is per chunk, not per token — so the KV
  at positions ``>= actual_end`` is whatever the pad tokens produced. It is the reader's job not to
  look there, and :attr:`PrefillRuntime.filled` reports where the real data ends.
* Chunks must arrive in order, per slot, because chunk N attends the prefix chunks 0..N-1 left in
  the cache. A gap or a repeat is rejected rather than absorbed.
* A short (padded) chunk ends the sequence: the next chunk would start mid-chunk, which the
  alignment rule above already forbids, so it is refused with the reason rather than the symptom.
"""

from dataclasses import dataclass

import torch
from loguru import logger

from models.demos.mistral_medium_3_5_128b.tt.model import shard_tokens


@dataclass(frozen=True)
class RuntimeConfig:
    """The chunk schedule's knobs. The engine's ``PrefillRunParams`` fields this runtime reads.

    ``first_layer_idx`` / ``is_first_rank`` / ``is_last_rank`` describe a pipeline rank's slice.
    This bring-up runs one rank holding the whole stack — the defaults — but they are part of the
    contract the engine reads off ``runtime.config``, so they are named here rather than assumed.
    """

    chunk_size: int
    max_seq_len: int
    num_layers: int
    num_users: int = 1
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True

    def __post_init__(self):
        assert (
            self.chunk_size > 0 and self.chunk_size % 32 == 0
        ), f"chunk_size ({self.chunk_size}) must be a positive multiple of the tile height"
        assert (
            self.max_seq_len >= self.chunk_size
        ), f"max_seq_len ({self.max_seq_len}) is below one chunk ({self.chunk_size})"
        assert self.num_users >= 1 and self.num_layers >= 1


class PrefillRuntime:
    """Drives :class:`~.model.MistralModel` one chunk at a time, with the schedule asserted.

    Built around an already-constructed model rather than constructing one: the model's weight load
    is minutes and its memory footprint is the binding constraint on this mesh, so a bring-up wants
    to choose when that happens. The engine's ``build_runtime`` would do both in one call.
    """

    def __init__(self, model, config: RuntimeConfig):
        assert model.chunk_size == config.chunk_size, (
            f"the model's block-cyclic cache period ({model.chunk_size}) must equal the runtime's "
            f"chunk_size ({config.chunk_size}); they address the same cache from opposite sides"
        )
        assert (
            model.num_layers == config.num_layers
        ), f"model has {model.num_layers} layers, runtime config says {config.num_layers}"
        self.model = model
        self.config = config
        self.mesh_device = model.mesh_device
        #: Real (non-pad) tokens written per slot so far — ``{slot_id: actual_end}``. The next
        #: chunk's ``actual_start`` is checked against this, and it is what a reader should trust
        #: over the cache's own capacity.
        self.filled: dict[int, int] = {}
        self._compiled = False

    # -- the engine's three calls -------------------------------------------------------------

    def compile(self, kv_cache) -> None:
        """Run one throwaway chunk so the per-chunk loop pays no first-call compile cost.

        Idempotent. The warm-up writes a chunk of KV at ``[0, chunk_size)`` of slot 0 — deliberately
        the same positions the first real chunk of slot 0 overwrites with the same op, so the only
        way this is observable is on a slot whose first chunk never runs. Callers that prefill slot
        0 from position 0 (every caller in this package) are unaffected; the alternative, warming up
        on a slot outside ``num_users``, would write outside the allocated cache.

        The bookkeeping is rolled back afterwards, so a warm-up does not consume slot 0's chunk 0.
        """
        if self._compiled:
            return
        tokens = self.make_chunk_input([0] * self.config.chunk_size)
        self.prefill_chunk(tokens, kv_cache, slot_id=0, actual_start=0, actual_end=self.config.chunk_size)
        self.filled.pop(0, None)
        self._compiled = True
        logger.info(f"[runtime] compiled on a {self.config.chunk_size}-token warm-up chunk")

    def make_chunk_input(self, token_ids):
        """One chunk's token ids as the device tensor :meth:`prefill_chunk` takes.

        Accepts fewer than ``chunk_size`` ids and pads with token 0 — the last chunk of a sequence
        that is not a whole multiple of the chunk size. The pad tokens are computed and their KV is
        written; ``actual_end`` is what says they are not real. Padding here rather than at the call
        site keeps the device shape static, which is what the engine's chunk loop (and a future
        trace capture) needs.
        """
        assert self.config.is_first_rank, (
            "make_chunk_input builds a token input, which only the first pipeline rank consumes; a "
            "later rank needs a placeholder activation, and multi-rank pipelining is the serving "
            "stage's concern (out of scope for this bring-up)"
        )
        ids = list(token_ids)
        assert (
            0 < len(ids) <= self.config.chunk_size
        ), f"a chunk holds 1..{self.config.chunk_size} token ids, got {len(ids)}"
        if len(ids) < self.config.chunk_size:
            ids = ids + [0] * (self.config.chunk_size - len(ids))
        return shard_tokens(
            self.mesh_device, self.model.mesh_config, torch.tensor(ids, dtype=torch.int32).reshape(1, -1)
        )

    def prefill_chunk(self, input_tensor, kv_cache, *, slot_id, actual_start, actual_end, request_id=0):
        """Prefill one chunk into slot ``slot_id``'s slice of ``kv_cache``.

        Returns ``None`` on the last (here: only) rank — the populated cache is the output, and the
        final hidden state of a prefill chunk is read by nobody, so it is freed here rather than
        handed back for the caller to remember to free.
        """
        cfg = self.config
        assert 0 <= slot_id < cfg.num_users, f"slot_id {slot_id} outside the {cfg.num_users} allocated user slots"
        assert 0 <= actual_start < actual_end, f"empty or negative chunk range [{actual_start}, {actual_end})"
        assert actual_end - actual_start <= cfg.chunk_size, (
            f"chunk [{actual_start}, {actual_end}) is {actual_end - actual_start} tokens, longer than "
            f"chunk_size ({cfg.chunk_size})"
        )
        assert (
            actual_end <= cfg.max_seq_len
        ), f"chunk ends at {actual_end}, past the cache's {cfg.max_seq_len}-token capacity"
        # Three overlapping ways to be out of order, checked from the most specific cause to the
        # least so the message names the mistake and not a consequence of it: a caller that kept
        # going past a padded tail is *also* misaligned and *also* mid-slot, but "the sequence
        # ended" is the only one of the three it can act on.
        expected = self.filled.get(slot_id, 0)
        assert expected % cfg.chunk_size == 0, (
            f"slot {slot_id} was last written a padded chunk ending at {expected}, which ended the "
            f"sequence; reset the slot to prefill it again"
        )
        assert actual_start % cfg.chunk_size == 0, (
            f"actual_start ({actual_start}) must be a multiple of chunk_size ({cfg.chunk_size}): the "
            f"cache's per-chip write is contiguous only at chunk boundaries"
        )
        assert actual_start == expected, (
            f"slot {slot_id} holds {expected} tokens, so the next chunk starts there, not at "
            f"{actual_start}: chunks must arrive in order and without gaps (chunk N attends 0..N-1)"
        )

        hidden = self.model(
            input_tensor,
            kv_cache=kv_cache,
            user_id=slot_id,
            cached_len=actual_start,
            want_logits=False,
        )
        self.filled[slot_id] = actual_end
        logger.debug(f"[runtime] slot {slot_id} chunk [{actual_start}, {actual_end}) req {request_id}")
        if cfg.is_last_rank:
            hidden.deallocate(True)
            return None
        return hidden

    # -- bookkeeping ---------------------------------------------------------------------------

    def reset_slot(self, slot_id: int) -> None:
        """Forget what slot ``slot_id`` holds, so it can be prefilled from position 0 again.

        The cache is not cleared: the next chunk 0 overwrites ``[0, chunk_size)`` and ``actual_end``
        bounds what anyone reads, so zeroing would be several GB of writes to hide nothing.
        """
        self.filled.pop(slot_id, None)

    def num_chunks(self, seq_len: int) -> int:
        """Chunks a ``seq_len``-token sequence takes, counting a padded tail as one."""
        return -(-seq_len // self.config.chunk_size)
