# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The device DFlash target: :class:`TtTarget` wraps :class:`~...tt.model.Qwen36Model`.

It implements the :class:`~...reference.dflash.targets.SpeculativeTarget` protocol that
:func:`~...reference.dflash.generate.dflash_generate` drives, so the same loop runs against the
host reference (:class:`~...reference.dflash.targets.HFTarget`) and against the device.

The drafter has neither an input embedding nor an LM head of its own; it borrows the target's.
:class:`TtTarget` serves the embedding either as a host gather of a few rows or on the mesh, and runs
the LM head on the mesh, where the target already holds it.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

from models.tt_transformers.tt.common import get_block_size


def load_target_embedding(path: str) -> torch.Tensor:
    """Pull just ``embed_tokens.weight`` out of the target checkpoint (~2.5 GB bf16).

    The host drafter has no embedding of its own, so it needs this as a torch tensor. ``lm_head`` is
    not loaded here because the device already holds it (see :meth:`TtTarget.lm_head`).
    """
    import json

    from safetensors import safe_open

    index = os.path.join(path, "model.safetensors.index.json")
    assert os.path.exists(index), f"no safetensors index at {index}"
    weight_map = json.load(open(index))["weight_map"]
    key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    with safe_open(os.path.join(path, weight_map[key]), framework="pt") as f:
        return f.get_tensor(key)


class TtTarget:
    """Device :class:`Qwen36Model`, driven through its masked-bucket prefill path.

    Every forward — the prompt and each speculative block — is a masked-bucket prefill at an
    absolute ``chunk_start``. That path carries GDN state across an offset and writes paged KV, but
    ``chunk_start`` must be a multiple of the bucket size, not of the paged block size:
    ``paged_fill_cache`` writes the whole padded bucket starting at block ``chunk_start // 64``, so
    consecutive segments are spaced by the bucket, not by their ``valid_len``. There is no device
    primitive that writes a multi-token run at an arbitrary offset — ``paged_fill_cache`` starts at a
    block boundary and ``paged_update_cache`` writes one token per batch element.

    Speculation advances by 1..16 tokens a step, so ``start`` is arbitrary and cannot be used as
    ``chunk_start`` directly. This class anchors instead: it keeps a bucket-aligned ``anchor`` with a
    GDN snapshot, and every forward re-runs the whole span ``[anchor, start + S)`` as one bucket at
    ``chunk_start=anchor``. Re-running already-computed tokens is free, since the bucket costs the
    same number of positions either way.

    Anchoring also removes rollback. Each forward restores GDN to the anchor and rewrites the entire
    ``[anchor, anchor + ANCHOR)`` KV span, so a rejected block is simply overwritten; there is nothing
    to undo and no replay. That is why :attr:`replays_after_rollback` is False here and True for
    :class:`HFTarget`.

    The caller must respect :meth:`max_block`: a block may not cross the anchor's bucket boundary, so
    near one the loop drafts a shorter block.
    """

    #: Masked-bucket size and the required chunk_start alignment. DFLASH_ANCHOR overrides it.
    #:
    #: A traced verify replays the whole bucket whatever the real span is (a trace bakes shapes), so
    #: verify cost scales with this width. A smaller bucket in turn makes max_block() truncate blocks
    #: more often and makes anchor crossings more frequent; see :meth:`anchor_for`.
    ANCHOR = int(os.environ.get("DFLASH_ANCHOR", "128"))
    #: Every forward recomputes from the anchor, so a rejected block needs no replay.
    replays_after_rollback = False

    #: Replay the verify trace past the first bucket as well as inside it. Off by default because it
    #: is only safe when the drafter keeps its KV history at stable addresses: a buffer reallocated
    #: under a parked trace corrupts the replay. Set it only when the drafter was built with
    #: ``ctx_capacity``; the growing-history drafter reallocates its history every step.
    allow_trace_past_anchor = False

    #: Greedy verification uses the verify logits only for ``argmax``, so the reduction can run on
    #: the mesh and only the winning ids cross PCIe instead of a ``[rows, vocab]`` logits readback.
    #: Honoured only on the traced narrow-head path and only for greedy callers;
    #: ``DFLASH_VERIFY_POSTERIOR=0`` opts out. See Qwen36Model._posterior_device.
    device_posterior = os.environ.get("DFLASH_VERIFY_POSTERIOR", "1") != "0"

    def __init__(
        self,
        model,
        tap_layer_ids,
        page_table,
        *,
        checkpoint_path=None,
        block_size=64,
        device_taps=False,
        anchor=None,
    ):
        self.model = model
        self.tap_layer_ids = list(tap_layer_ids)
        self.page_table = page_table
        # Per-request anchor, overriding ANCHOR; see anchor_for() for how a caller sizes it.
        if anchor is not None:
            assert anchor % 64 == 0, f"anchor {anchor} must be a multiple of the 64-row page"
            self.ANCHOR = int(anchor)
        self.capacity = page_table.shape[1] * block_size
        # device_taps: keep the residual taps on the mesh for a ttnn drafter instead of reading them
        # back to host.
        self.device_taps = device_taps
        model.set_residual_taps(self.tap_layer_ids, keep_on_device=device_taps)

        # Loaded lazily, and only the embedding: the LM head stays on the mesh.
        self._checkpoint_path = checkpoint_path
        self._embed = None
        self._anchor = 0
        self._anchor_gdn = None
        self._tokens = torch.zeros(1, self.capacity + self.ANCHOR, dtype=torch.long)

    @property
    def hidden_size(self) -> int:
        return self.model.args.dim

    def max_block(self, start: int) -> int:
        """Largest block that fits without crossing the anchor's bucket boundary."""
        return self.ANCHOR - (start % self.ANCHOR)

    #: Cost-model constants for :meth:`anchor_for`.
    #: Verify cost per padded row of the bucket, ms (verify time grows roughly linearly in width).
    VERIFY_MS_PER_ROW = 0.25
    #: Cost of one anchor crossing, ms: an eager whole-bucket forward, the trace re-capture it
    #: forces, and the bucket's eager tail steps. Roughly independent of the bucket width.
    CROSSING_MS = 4000.0
    #: Tokens committed per speculative step, used only to turn a token budget into a step count.
    #: Deliberately on the low side: underestimating acceptance overestimates the steps and so
    #: widens less, which is the safe direction.
    ASSUMED_ACCEPTANCE = 4.0

    @staticmethod
    def crossings_for(total_tokens, anchor) -> int:
        """How many bucket boundaries a request of ``total_tokens`` crosses at this ``anchor``.

        Boundaries sit at ``anchor``, ``2*anchor``, ...; one landing exactly on the last token is
        never reached, so this is ``(total_tokens - 1) // anchor``.
        """
        return max(0, (int(total_tokens) - 1) // int(anchor))

    @classmethod
    def anchor_for(cls, total_tokens, cap=512, floor=128, page=64, new_tokens=None, acceptance=None):
        """The bucket width with the lowest predicted cost for this request.

        Two terms trade off. Each anchor crossing costs an eager whole-bucket forward plus a trace
        re-capture (``CROSSING_MS``); each padded row costs ``VERIFY_MS_PER_ROW`` on every step. So
        widening by N rows to avoid C crossings pays while ``steps * VERIFY_MS_PER_ROW * N <
        CROSSING_MS * C``. Short requests usually want the smallest bucket that holds them (no
        crossings); long requests may prefer a narrower bucket that only reduces the crossings. The
        argmin is taken over every candidate width from ``floor`` to the smallest bucket that fits.

        The constants were fitted on requests of a few hundred tokens; predictions for much longer
        requests are extrapolated.

        Args:
            total_tokens: prompt + generation. Sets which buckets fit and how many boundaries lie
                inside the request.
            cap: the widest bucket to consider. The demo's ``trace_region_size`` is sized for 512;
                raising it needs a bigger trace region.
            new_tokens: the generation budget, which is what the per-step cost is paid over. Left
                None it falls back to ``total_tokens``, which overestimates the steps and so widens
                less.
            acceptance: tokens per step, defaulting to :attr:`ASSUMED_ACCEPTANCE`.
        """
        total = int(total_tokens)
        floor, page, cap = int(floor), int(page), int(cap)
        # Never wider than the smallest bucket that holds the request: past that there is nothing
        # left to buy, and every extra row is paid for on every step.
        want = min(cap, max(floor, -(-total // page) * page))
        steps = max(
            1, round((total if new_tokens is None else int(new_tokens)) / (acceptance or cls.ASSUMED_ACCEPTANCE))
        )

        def cost(anchor):
            # Relative to `floor`, which drops out of the comparison: only the differences matter.
            widening = steps * cls.VERIFY_MS_PER_ROW * (anchor - floor)
            return widening + cls.CROSSING_MS * cls.crossings_for(total, anchor)

        # Ties go to the narrower bucket (min is stable), which is the cheaper one to capture.
        return min(range(floor, want + 1, page), key=cost)

    def reset(self) -> None:
        # _has_generated deliberately survives reset(): it tracks whether this process has compiled
        # the loop's programs, which a sequence reset does not undo.
        self.model._reset_gdn_state_for_new_sequence()
        self._anchor = 0
        self._anchor_gdn = self.model.save_gdn_state()
        self._tokens.zero_()

    def _taps_cat(self, parts):
        """Join per-bucket taps. Host taps concat on the row axis; device taps concat per tap."""
        if len(parts) == 1:
            return parts[0]
        if not self.device_taps:
            return torch.cat(parts, dim=1)
        import ttnn

        return [
            ttnn.concat([p[j] for p in parts], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for j in range(len(parts[0]))
        ]

    def taps_head(self, taps, rows: int):
        """The first ``rows`` rows — the accepted prefix of a block's taps."""
        if not self.device_taps:
            return taps[:, :rows]
        import ttnn

        return [
            (
                t
                if t.shape[-2] == rows
                else ttnn.slice(t, (0, 0, 0, 0), (1, 1, rows, t.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            for t in taps
        ]

    def taps_tail(self, taps, rows: int):
        """The trailing ``rows`` rows of a tap set — a block forward's own positions inside a bucket."""
        if not self.device_taps:
            return taps[:, -rows:]
        import ttnn

        out = []
        for t in taps:
            have = t.shape[-2]
            out.append(
                t
                if have == rows
                else ttnn.slice(
                    t, (0, 0, have - rows, 0), (1, 1, have, t.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )
        return out

    def embed_device(self, ids):
        """The target's input embedding of ``ids``, on the mesh, replicated at full hidden width.

        ``model.embd`` returns the embedding fractured on the hidden dim (TP), so it is gathered back
        to full width for the drafter, which keeps its hidden replicated.
        """
        import ttnn

        seq = ids.shape[1]
        multi = self.model.num_devices > 1
        tok = ttnn.from_torch(
            ids.to(torch.int32).cpu(),
            dtype=ttnn.uint32,
            device=self.model.device,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device)) if multi else {}),
        )
        return self._embed_tokens(tok, seq, own=True)

    def _embed_tokens(self, tok, seq, *, own):
        import ttnn
        from models.tt_transformers.tt.ccl import tt_all_gather

        multi = self.model.num_devices > 1
        x = self.model.embd(tok)
        if own:
            ttnn.deallocate(tok)
        x = ttnn.reshape(x, (1, 1, seq, x.shape[-1]))
        if multi:
            x = tt_all_gather(
                x,
                self.model.device,
                self.model.tt_ccl,
                cluster_axis=None,
                dim=3,
                topology=ttnn.Topology.Linear,
                num_workers_per_link=2,
                chunks_per_sync=10,
            )
        return x

    def draft_ids_device(self, hidden, *, keep_rows):
        """Greedy draft token ids, argmaxed on device -- ``keep_rows`` ints cross PCIe, not logits.

        Under greedy decoding the host only ever argmaxes the draft logits, so the argmax runs on
        the device and the transfer is a handful of uint32s instead of ``[rows, vocab]`` logits.

        This is one op, not a reduction tree: ``Qwen36Model._lm_head`` all-gathers its vocab shards
        before returning, so every device holds the full-width row and device 0's argmax is the
        global one.

        Sampling still needs the distribution, so ``temperature > 0`` keeps :meth:`lm_head_device`.
        """
        import ttnn

        logits = self.model._lm_head(hidden)
        rows, width = logits.shape[-2], logits.shape[-1]
        vocab = self.model.vocab_size
        owned = []

        keep = logits
        if keep_rows is not None and keep_rows < rows:
            keep = ttnn.slice(keep, (0, 0, rows - keep_rows, 0), (1, 1, rows, width))
            owned.append(keep)
        if width > vocab:
            # Must trim before the argmax: the head's output is padded past vocab_size, and an
            # argmax over the padded columns can silently return an index that is not a token.
            keep = ttnn.slice(keep, (0, 0, 0, 0), (1, 1, keep.shape[-2], vocab))
            owned.append(keep)

        # Argmax in ROW_MAJOR, not TILE: ttnn.argmax over a vocab-wide row is far slower in TILE
        # layout, even counting the untilize.
        rm = ttnn.to_layout(keep, ttnn.ROW_MAJOR_LAYOUT)
        ids = ttnn.argmax(rm, dim=-1)
        ttnn.deallocate(rm)
        if self.model.num_devices > 1:
            host = ttnn.to_torch(ttnn.get_device_tensors(ids)[0])
        else:
            host = ttnn.to_torch(ids)
        # Never test membership with `in` here: `x in owned` calls `==`, which on ttnn.Tensor is an
        # elementwise device op and fails against already-freed tensors. `owned` only ever holds
        # slices this function made, so `logits` is not in it by construction.
        for t in owned:
            ttnn.deallocate(t)
        ttnn.deallocate(logits)
        ttnn.deallocate(ids)
        return host.reshape(1, -1)[:, -keep_rows:].long()

    def lm_head_device(self, hidden, *, keep_rows=None):
        """Draft logits for a device hidden tensor, via the mesh-resident LM head.

        The drafter borrows the target's head, and on this backend that head already lives on the
        mesh (``Qwen36Model.lm_head_weight``, vocab-sharded with a full-width replicated input —
        exactly the shape the drafter's hidden states have). ``keep_rows`` returns only the trailing
        rows, which is what the drafted slots are.

        :meth:`lm_head` is the same thing for a host tensor.
        """
        import ttnn

        logits = self.model._lm_head(hidden)
        # Slice on device, then read one device's copy (as Qwen36Model._read_verify_logits does):
        # the LM head all-gathers its vocab shards, so every device already holds the full row.
        # `keep_rows` trims the row axis: slot 0 of a block is the confirmed anchor, so only the
        # trailing q_len-1 rows are ever read.
        keep, sliced = logits, None
        rows = logits.shape[-2]
        if keep_rows is not None and keep_rows < rows:
            sliced = ttnn.slice(logits, (0, 0, rows - keep_rows, 0), (1, 1, rows, logits.shape[-1]))
            keep = sliced
        if self.model.num_devices > 1:
            host = ttnn.to_torch(ttnn.get_device_tensors(keep)[0])
        else:
            host = ttnn.to_torch(keep)
        if sliced is not None:
            ttnn.deallocate(sliced)
        ttnn.deallocate(logits)
        host = host.reshape(-1, host.shape[-1])[:, : self.model.vocab_size].float()
        # Already trimmed on device when keep_rows applied; the tail slice is now a no-op guard.
        return (host if keep_rows is None else host[-keep_rows:]).unsqueeze(0)

    def _run(self, lo: int, hi: int, keep_rows=None, posterior=False):
        """Run ``[lo, hi)`` from the anchor's GDN state as one bucket at ``chunk_start=lo``.

        ``keep_rows`` is how many trailing rows of logits the caller will actually use. ``None``
        means all of them. ``0`` means none, and the LM head is skipped entirely -- :meth:`forward`'s
        whole-bucket iterations pass 0, because ``max_block`` keeps a speculative block inside the
        tail bucket and so their logits are discarded in full.

        Returns ``(logits, taps, got_posterior)``. With ``posterior=True`` the caller is asking for
        the greedy argmax of those rows instead of the rows themselves, reduced on device; only the
        traced narrow-head path can serve that, so ``got_posterior`` says whether it did. Every
        other path returns logits and leaves the argmax to the caller, which is the same answer by
        a slower route -- so the flag is an optimization, never a correctness switch.
        """
        assert lo % self.ANCHOR == 0, f"chunk_start {lo} is not {self.ANCHOR}-aligned"
        length = hi - lo
        assert 1 <= length <= self.ANCHOR, f"span [{lo}, {hi}) does not fit one {self.ANCHOR} bucket"
        # The GDN restore runs on host, outside the trace: its snapshot buffers are retaken by
        # reset() between generations, so addresses baked into a capture would go stale.
        self.model.restore_gdn_state(self._anchor_gdn)
        # length == ANCHOR is a whole bucket (prompt prefill), and gdn/tp.py::_normalize_valid_len
        # turns valid_len >= T into None -- masking skipped, different programs than the capture.
        # The trace serves partial buckets only; a full one falls back to eager.
        #
        # The trace is captured at chunk_start=0 (capture_verify_trace). One capture serves every
        # valid_len below the bucket, but replaying it at a later chunk_start is only correct when
        # nothing it bakes has been reallocated since, so by default only lo == 0 is traced and
        # later buckets run eagerly. allow_trace_past_anchor / DFLASH_TRACE_PAST_ANCHOR=1 lifts that
        # restriction; it requires a drafter with stable KV addresses (ctx_capacity) and relies on
        # the re-capture in _recapture_after_anchor.
        #
        # TODO: stage whatever the capture still bakes from chunk_start (the int `chunk_start_idx`
        # passed alongside the staged `chunk_start_idx_tensor` in tt/model.py,
        # _forward_prefill_chunk_masked_tp) so one capture serves every offset without re-capture.
        _past_anchor = self.allow_trace_past_anchor or os.environ.get("DFLASH_TRACE_PAST_ANCHOR") == "1"
        if getattr(self, "_traced_verify", False) and length < self.ANCHOR and (lo == 0 or _past_anchor):
            # Trace replay instead of eager dispatch; same all-row logits as the eager entry point.
            # One capture serves every `length` below the bucket -- valid_len lives in the staged
            # GDN mask's contents.
            token_buf = torch.zeros(1, self.ANCHOR, dtype=self._tokens.dtype)
            token_buf[:, :length] = self._tokens[:, lo:hi]
            # The device-side argmax is only used with the narrow head. See
            # Qwen36Model._posterior_device.
            want_posterior = bool(posterior) and getattr(self.model, "_vt_narrow_head", False)
            logits = self.model.verify_traced(
                token_buf, length, lo, self.page_table, self.ANCHOR, keep_rows=keep_rows, posterior=want_posterior
            )
            if want_posterior:
                return logits, self.model.take_taps(length), True
        else:
            logits = self.model.prefill_block_all_logits(
                self._tokens[:, lo:hi],
                self.page_table,
                actual_len=length,
                chunk_start=lo,
                bucket=self.ANCHOR,
                keep_rows=keep_rows,
            )
        return logits, self.model.take_taps(length), False

    def _recapture_after_anchor(self):
        """Re-take the verify trace when the anchor advances.

        Required whenever the trace is used past an anchor, so it is enabled with that path rather
        than behind its own switch.

        Crossing an anchor runs a whole-bucket forward (``length == ANCHOR``), which ``_run`` sends
        down the eager fallback. That eager forward allocates while the trace is parked and can land
        on memory the trace's working set (not just its output) was captured into, so later replays
        compute from memory they no longer own. Re-capturing after the crossing restores a valid
        trace; the only alternative would be never running an eager forward with a trace parked.
        """
        if not getattr(self, "_traced_verify", False):
            return
        # Only the past-anchor traced path needs this: with the eager fallback the trace is never
        # replayed after a crossing, so there is nothing to keep alive.
        if not (self.allow_trace_past_anchor or os.environ.get("DFLASH_TRACE_PAST_ANCHOR") == "1"):
            return
        import time as _time

        _t0 = _time.perf_counter()
        self.model.release_verify_trace()
        self.model.capture_verify_trace(
            self._capture_page_table(),
            self.ANCHOR,
            narrow_head=getattr(self, "_narrow_head", False),
        )
        self._recaptures = getattr(self, "_recaptures", 0) + 1
        logger.info(
            f"[dflash] re-captured verify trace at anchor {self._anchor} "
            f"in {_time.perf_counter() - _t0:.2f}s (#{self._recaptures})"
        )

    def _capture_page_table(self):
        """Spare pages for the capture's throwaway forwards. See :meth:`_recapture_after_anchor`.

        A capture is not a dry run: it executes a warm-up forward, the narrow-head probe and the
        captured body itself, each over ``bucket`` zero tokens at ``chunk_start=0``, and each ends
        in a ``paged_fill_cache``. Handed the live page table those writes would land on the pages
        holding the bucket that just completed, zeroing KV history later steps attend over.

        So point the capture at the table's last pages instead. ``verify_traced`` re-stages the real
        table on every replay (``stage_verify_inputs`` rewrites ``_vt_full_pt``/``_vt_chunk_pt``),
        so no part of the real mapping is baked into the trace. If the sequence has grown far enough
        to reach those pages there is nowhere safe to put them, and the live table is returned rather
        than aliasing a page in use.
        """
        return self.capture_page_table(
            self.page_table, self._anchor, get_block_size(self.model._paged_kv_caches), self.ANCHOR
        )

    @staticmethod
    def capture_page_table(page_table, anchor, block_size, bucket):
        """The page table a capture should run against: the bucket remapped onto the last pages.

        Split out from :meth:`_capture_page_table` so the arithmetic is testable without a mesh. The
        capture writes ``bucket`` rows starting at ``chunk_start=0``, which ``paged_fill_cache`` maps
        onto the first ``bucket // block_size`` entries -- so those are the entries to redirect, and
        only those.

        Returns the table unchanged when the sequence has grown close enough to the end that the
        spare pages are no longer spare: the degraded case is "no protection", never "wrong pages".
        """
        n_blocks = page_table.shape[1]
        per_bucket = bucket // block_size
        # Pages the sequence can still reach, with one bucket of headroom.
        live = -(-(anchor + 2 * bucket) // block_size)
        if n_blocks - per_bucket < live:
            return page_table
        scratch = page_table.clone()
        scratch[:, :per_bucket] = page_table[:, n_blocks - per_bucket :]
        return scratch

    def enable_traced_verify(self, warm_tokens=None, narrow_head=False):
        """Capture the verify trace and route :meth:`_run` through it.

        Off by default; a capture costs trace memory. ``release_verify_trace`` on the model undoes it.

        Call this only after at least one full eager generation. ``capture_verify_trace`` warms only
        the programs its own dummy forward touches; the speculative loop needs more (the drafter's
        projections, the tap gather, the LM head at the drafted width, the whole-bucket eager
        fallback ``_run`` takes when ``length == ANCHOR``). Compiling those with a trace parked does
        not raise -- it hangs the process and can wedge the device -- so the assert below turns that
        into an immediate error.
        """
        assert self._anchor_gdn is not None, "call reset() before enabling the traced verify"
        assert getattr(self, "_has_generated", False), (
            "enable_traced_verify() requires at least one completed EAGER generation first -- "
            "capture_verify_trace only warms its own forward, and compiling the rest with a trace "
            "parked hangs the process and wedges the device. Run one dflash_generate() (8 tokens is "
            "enough) before enabling the trace."
        )
        # narrow_head: capture stops at the norm and the LM head runs per replay over a 32- or
        # 64-row tile-aligned window instead of all ANCHOR rows.
        self.model.capture_verify_trace(
            self.page_table,
            self.ANCHOR,
            warm_tokens=warm_tokens,
            narrow_head=narrow_head,
            # The device-side verify argmax runs outside the trace, per replay, so its programs are
            # first reached after the capture unless they are warmed with the head widths.
            warm_posterior=narrow_head and self.device_posterior,
        )
        self._traced_verify = True
        self._narrow_head = narrow_head

    def forward(self, ids, start, *, all_logits=True, posterior=False):
        """Run ``ids`` at absolute ``start``; ``all_logits`` is ignored (a bucket computes all rows).

        A long prompt is fed as consecutive whole buckets, each of which also re-anchors, so prompts
        are not limited to one bucket — only a single speculative block is, by :meth:`max_block`.

        ``posterior=True`` asks for the greedy argmax of the returned rows -- ``[1, S]`` int64 --
        instead of the rows themselves, computed on device where the path allows it and on host
        where it does not. It is the same answer either way; only greedy callers may ask, because
        sampling needs the distribution (:meth:`device_posterior` gates it).
        """
        # enable_traced_verify() checks this to refuse a capture on a target whose programs have
        # never been compiled.
        self._has_generated = True
        S = ids.shape[1]
        end = start + S
        assert end <= self.capacity, f"position {end} exceeds the {self.capacity}-token page table"
        assert self._anchor_gdn is not None, "call reset() before the first forward"
        assert start <= self._anchor + self.ANCHOR, (
            f"start {start} skips past the anchor at {self._anchor}; a block must not cross a bucket "
            "boundary — see max_block()"
        )
        self._tokens[:, start:end] = ids.to(self._tokens.dtype).cpu()

        first_anchor = self._anchor
        logit_parts, tap_parts = [], []

        # How many trailing logit rows this call will actually return. Only these are worth
        # computing: the LM head and its vocab all-gather are sized by the bucket while `want` is at
        # most the block -- and everything else is discarded below.
        want = S if all_logits else 1
        # Does `want` fit inside the tail bucket alone? It does for every speculative step, because
        # max_block() stops a block crossing a bucket boundary, and for any all_logits=False call.
        # A long all_logits=True prompt is the exception: it spans several buckets and genuinely
        # needs every row, so that case keeps the whole-bucket behaviour.
        tail_len = end - (first_anchor + self.ANCHOR * ((end - first_anchor - 1) // self.ANCHOR))
        narrow = want <= tail_len

        # Whole buckets: all-real and already committed, so each one also re-anchors the snapshot.
        crossed = False
        while end - self._anchor > self.ANCHOR:
            # keep_rows=0 when narrow: these buckets' logits are discarded in full by the slice at
            # the end of this method, so the head never needs to run on them.
            lg, tp, _ = self._run(self._anchor, self._anchor + self.ANCHOR, keep_rows=0 if narrow else None)
            logit_parts.append(lg)
            tap_parts.append(tp)
            self._anchor += self.ANCHOR
            # Reuse the snapshot's buffers rather than reallocating every bucket.
            self._anchor_gdn = self.model.save_gdn_state(into=self._anchor_gdn)
            crossed = True
        # Re-capture once, after the last crossing: every forward in the loop above is a whole
        # bucket and runs eagerly, so nothing replays the trace until the tail below. A long prompt
        # crosses many buckets in one call and pays one re-capture, not one per bucket. Order
        # matters: the snapshot must be taken before the re-capture, because a capture runs real
        # forwards that advance all 48 GDN recurrent states and every later _run() restores from the
        # snapshot. See _recapture_after_anchor.
        if crossed:
            self._recapture_after_anchor()
        # The partial tail bucket — where a speculative block always lands. A device-side argmax is
        # only askable when the answer is these rows alone, which `narrow` is exactly the test for.
        lg, tp, got_posterior = self._run(
            self._anchor, end, keep_rows=want if narrow else None, posterior=posterior and narrow
        )
        logit_parts.append(lg)
        tap_parts.append(tp)

        if narrow:
            # `lg` is already exactly the trailing `want` rows; the earlier parts are all None.
            logits = lg
        else:
            logits = torch.cat(logit_parts, dim=1) if len(logit_parts) > 1 else logit_parts[0]
            logits = logits[:, -want:]
        if posterior and not got_posterior:
            # The path could not reduce on device (wide head, eager fallback, or a multi-bucket
            # prompt). Same answer, one host argmax later.
            logits = torch.argmax(logits, dim=-1)
        return logits, self.taps_tail(self._taps_cat(tap_parts), S)

    def snapshot(self):
        """Nothing to capture: every forward recomputes from the anchor."""
        return None

    def restore(self, snap, length):
        """No-op — see the class docstring. The next forward rewrites the whole bucket."""

    def embed(self, ids):
        """Host gather from the target's embedding table — a handful of rows, so host is fine."""
        if self._embed is None:
            from models.demos.blackhole.qwen36.tt.dflash.config import resolve_target_path

            self._embed = load_target_embedding(self._checkpoint_path or resolve_target_path())
        return torch.nn.functional.embedding(ids.cpu(), self._embed)

    def lm_head(self, hidden):
        """Draft logits for a host hidden tensor: upload, then :meth:`lm_head_device`.

        Uploading the hidden states and reading the logits back avoids a 5120 x vocab CPU matmul,
        which is why the host drafter routes through here too.
        """
        import ttnn

        seq = hidden.shape[-2]
        multi = self.model.num_devices > 1
        x = ttnn.from_torch(
            hidden.reshape(1, 1, seq, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.model.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device)) if multi else {}),
        )
        out = self.lm_head_device(x)
        ttnn.deallocate(x)
        return out
