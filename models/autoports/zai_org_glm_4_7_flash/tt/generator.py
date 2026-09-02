# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving-shaped generator for the TTNN GLM-4.7-Flash full model (one Blackhole chip).

Implements ``models.common.readiness_check.contract.Generator`` and the
canonical split-sampling traced-decode contract from ``$tt-enable-tracing``.

Two API levels
==============

low level (cache/page-table/positions owned by the caller; what a vLLM adapter
drives)
    ``prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...)``
    ``decode_forward(tokens, start_pos, *, page_table, kv_cache, ...)``
    plus explicit state plumbing: :meth:`allocate_kv_cache`,
    :meth:`bind_decode_state`, :meth:`capture_decode_trace`,
    :meth:`decode_step_traced`, :meth:`set_sampling_params`.

high level (the generator owns cache, page table and decode state)
    ``prefill_logits(prompt_token_ids)``
    ``generate(prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, ...)``
    ``reset()``

Split sampling
==============

Token-out decode is two cooperating traces over persistent device tensors:

1. the *model* decode trace runs embedding -> 47 layers -> final norm -> LM head,
   derives the RoPE index from the current position on device on the way in and
   advances that position with ``ttnn.plus_one(..., skip_negative_entries=True)``
   on the way out, returning sampler-ready logits ``[1, 1, 32, 154880]``;
2. the *sampling* trace is ``models.common.sampling.SamplingGenerator``, captured
   over that exact logits tensor with ``tt_out_tok`` pointing at the persistent
   decode token tensor, so the sampled token becomes the next decode input
   without ever reaching the host.

Steady-state decode is therefore: replay model trace, replay sampling trace,
read one uint32 word. No host argmax, no full-logits readback, no per-token
token/position/page-table refresh (and no RoPE-index tensor to refresh at all).
:attr:`counters` records exactly that and the tests assert it.

Sampler choice
==============

``models/common/sampling`` (``SamplingGenerator`` + ``TTSampling``) is used;
``models/common/modules/sampling/sampling_1d.py`` was rejected. The deciding
reason is the single-device vocab split: this model's vocab is 154880 and
``TTSampling.num_single_device_vocab_splits`` picks 4 chunks of 38720, each
inside ``ttnn.topk``'s 65536 practical width, while ``Sampling1D`` hardcodes a
2-way split (``sampling_1d.py:570``) that would hand ``ttnn.topk`` a
77440-wide input. ``Sampling1D`` additionally has no greedy tie-break, no seed
management, and no ``tt_out_tok`` decode-feedback user in-tree. Full comparison
in ``doc/full_model/README.md``.

Greedy is *semantically greedy split sampling*, not force-argmax: the top-k
stage always gathers ``max_top_k`` = 32 candidates per vocab chunk (128 total,
Wt = 4, a power of two as ``ttnn.sampling`` requires) and the draw runs with
``k=1, p=0, temp=1``. Force-argmax was measured against it
(``probe/greedy_sampler_probe.py``: 1.084 ms traced vs 1.108 ms, both 32/32
against torch argmax) and left off deliberately: 0.024 ms is 0.1% of the
23.0 ms token-out step, while force-argmax is greedy-only and toggling it makes
``reset_sampling_params`` release every captured sampling trace, so a mixed
greedy/sampled workload would recapture on each mode change.

Host sampling
=============

``host_sampling=True`` (or ``build_generator(..., host_sampling=True)``) is an
explicit compatibility mode for tests that need host-side token selection: the
model trace still runs, but the logits are read back and argmaxed on the host.
It is never used by the measured token-out path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.model import DEFAULT_HF_MODEL_ID, GLM47FlashModel, resolve_checkpoint_dir

try:  # the readiness contract is optional at import time so the module stays usable standalone
    from models.common.readiness_check.contract import Generator as _ReadinessGenerator
except Exception:  # pragma: no cover - only when the harness is absent
    _ReadinessGenerator = object

from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

#: ``ttnn.sampling`` samples one row per user and ``TTSampling`` floors its
#: batch to a full tile, so the sampler always sees 32 logits rows.
SAMPLER_ROWS = 32

GREEDY = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)


class _SamplingArgs:
    """The ``args`` duck type ``TTSampling``/``TTPenalties`` read."""

    def __init__(self, *, vocab_size, padded_vocab_size, mesh_device, max_batch_size=SAMPLER_ROWS, max_top_k=32):
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.cluster_shape = tuple(int(d) for d in mesh_device.shape)
        self.max_batch_size = max_batch_size
        self.max_top_k = max_top_k
        self.sampling_dp = 1


def _new_counters() -> Dict[str, int]:
    return {
        "model_trace_replays": 0,
        "sampling_trace_replays": 0,
        "eager_decode_steps": 0,
        "token_input_refreshes": 0,
        "position_refreshes": 0,
        # No separate RoPE-index tensor exists any more: the model derives it from
        # the current position on device. Kept at 0 so the counter contract is stable.
        "rope_index_refreshes": 0,
        "page_table_refreshes": 0,
        "device_synchronizations": 0,
        "token_readbacks": 0,
        "full_logits_readbacks": 0,
        "host_argmax_calls": 0,
        "kv_cache_resets": 0,
        "trace_recaptures": 0,
    }


class GLM47FlashGenerator(_ReadinessGenerator):
    """Readiness/serving generator around :class:`GLM47FlashModel`."""

    def __init__(
        self,
        model: GLM47FlashModel,
        *,
        tokenizer=None,
        host_sampling: bool = False,
        enable_sampling: bool = True,
    ):
        self.model = model
        self.mesh_device = model.mesh_device
        self.max_batch_size = model.max_batch_size
        self.tokenizer = tokenizer
        self.host_sampling = host_sampling
        self.counters = _new_counters()

        self.sampling = None
        if enable_sampling:
            args = _SamplingArgs(
                vocab_size=model.vocab_size,
                padded_vocab_size=model.vocab_size,
                mesh_device=model.mesh_device,
            )
            self.sampling = SamplingGenerator(args=args, mesh_device=model.mesh_device, tt_ccl=None)
            self.set_sampling_params(GREEDY)

        # generator-owned state (high-level API)
        self._kv_cache = None
        self._page_table_torch = None

        # persistent decode trace state
        self._tokens_dev = None  # [1, 1, 1, 32] uint32 - also the sampler's tt_out_tok
        self._prefill_tokens_dev = None  # [1, 1, 1, 32] uint32 - prefill sampler output
        self._pos_dev = None  # [B] int32, -1 = inactive slot; the RoPE index is derived from it on device
        self._page_table_dev = None  # [B, blocks] int32
        self._page_table_caller_owned = False
        #: Host mirror of the device current-position tensor. The captured graph
        #: advances the device copy with ttnn.plus_one, so this is the only way
        #: the host can tell that the *next* replay would step past the context.
        self._host_positions = None
        self._bound_cache = None
        self._decode_trace_id = None
        self._decode_logits = None
        #: num_program_cache_entries() at capture time; see
        #: _maybe_recapture_after_compile.
        self._program_cache_entries_at_capture = None
        self._sampling_traced = False
        self._sampling_params = GREEDY

    # ------------------------------------------------------------------ construction helpers

    @property
    def max_seq_len(self) -> int:
        return self.model.max_seq_len

    def set_sampling_params(self, sampling_params: SamplingParams):
        """Push sampling params to the device. Changing between greedy and
        sampled params is fine; the sampling trace is keyed on the mode."""
        if self.sampling is None:
            self._sampling_params = sampling_params
            return
        formatted = format_sampling_params(sampling_params, self.sampling.tt_sampling.max_batch_size)
        self.sampling.reset_sampling_params(formatted)
        self._sampling_params = sampling_params

    # ------------------------------------------------------------------ cache / page table

    def allocate_kv_cache(self, dtype=None):
        """Allocate the per-layer paged latent caches (caller-owned form)."""
        return self.model.allocate_kv_cache(dtype=dtype)

    def default_page_table(self, batch=None):
        return self.model.default_page_table(batch)

    def _ensure_owned_state(self):
        """Allocate the generator-owned cache + page table for the high-level API.

        A caller that already bound its own cache through
        :meth:`bind_decode_state` keeps it: allocating a second full-context
        cache would cost another 5.4 GiB.
        """
        if self._kv_cache is not None:
            return
        if self._bound_cache is not None:
            # Adopt the caller's binding rather than allocating a second
            # full-context cache. Do NOT invent a page table: if the caller
            # owns a device one, _page_table_torch must stay None so a later
            # only_if_changed diff cannot compare against a table that was
            # never on the device.
            self._kv_cache = self._bound_cache
            return
        self._kv_cache = self.allocate_kv_cache()
        self._page_table_torch = self.default_page_table()
        self.bind_decode_state(kv_cache=self._kv_cache, page_table=self._page_table_torch)

    # ------------------------------------------------------------------ persistent decode state

    def bind_decode_state(self, *, kv_cache, page_table):
        """Allocate/point the persistent decode trace inputs at ``kv_cache``.

        Must run before :meth:`capture_decode_trace`; the captured trace binds
        to exactly these tensors and to these cache buffers. The decode batch is
        the model's ``max_batch_size`` and is not a parameter here: the decoder
        pins its per-slot shard grids at construction, so a narrower decode
        batch needs a differently-constructed model, not a differently-bound
        one.
        """
        import torch

        batch = self.max_batch_size
        if self._tokens_dev is None:
            self._tokens_dev = ttnn.from_torch(
                torch.zeros(1, 1, 1, SAMPLER_ROWS, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._pos_dev = ttnn.from_torch(
                torch.zeros(batch, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Separate sampler output for prefill: the wanted prompt position is
            # not always sampler row 0, so prefill must not write the decode
            # token buffer directly.
            self._prefill_tokens_dev = ttnn.from_torch(
                torch.zeros(1, 1, 1, SAMPLER_ROWS, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._decode_trace_id is not None:
            # The captured trace hardcodes the buffers bound here; silently
            # swapping them would make replay read the old cache/page table.
            same_cache = kv_cache is self._bound_cache
            if isinstance(page_table, ttnn.Tensor):
                same_pt = page_table is self._page_table_dev
            else:
                # A torch table is only "the same binding" when the generator
                # owns the device buffer it would be copied into; if the caller
                # owns it, rebinding would write into their tensor.
                same_pt = not self._page_table_caller_owned
            if not (same_cache and same_pt):
                raise RuntimeError(
                    "bind_decode_state cannot rebind after capture_decode_trace: the captured graph reads "
                    "the previously bound cache / page-table buffers. Call teardown() and recapture instead."
                )
        if isinstance(page_table, ttnn.Tensor):
            self._page_table_dev = page_table
            # The caller owns this buffer and updates it itself; we must not
            # diff against or write into it.
            self._page_table_torch = None
            self._page_table_caller_owned = True
        elif self._page_table_dev is None:
            self._page_table_dev = self.model.page_table_to_device(page_table)
            self._page_table_torch = torch.as_tensor(page_table, dtype=torch.int32)
            self._page_table_caller_owned = False
            self.counters["page_table_refreshes"] += 1
        else:
            self._page_table_caller_owned = False
            self.refresh_page_table(page_table)
        self._bound_cache = kv_cache

    def refresh_page_table(self, page_table_torch, *, only_if_changed: bool = False):
        """Copy a new page table into the persistent trace input. Call only
        when the page table actually changes; the steady-state decode loop
        performs no page-table copies."""
        import torch

        if self._page_table_caller_owned:
            raise RuntimeError(
                "refresh_page_table cannot write into a caller-owned device page table; update that tensor "
                "in place, or bind a torch page table so the generator owns the device copy."
            )
        pt = torch.as_tensor(page_table_torch, dtype=torch.int32)
        if only_if_changed and self._page_table_torch is not None:
            current = torch.as_tensor(self._page_table_torch, dtype=torch.int32)
            if current.shape == pt.shape and torch.equal(current, pt):
                return
        host = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._page_table_dev)
        self._page_table_torch = pt
        self.counters["page_table_refreshes"] += 1

    def set_decode_positions(self, positions):
        """Set the current-position trace input (request boundary only: inside a
        fixed-step decode loop the trace advances it itself).

        ``-1`` marks an inactive slot. There is no separate RoPE-index tensor:
        the model derives it from this one on device every step, so an inactive
        slot is pinned at RoPE index 0 instead of drifting.
        """
        import torch

        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        slots = int(self._pos_dev.shape[0])
        if pos.numel() > slots:
            raise ValueError(f"expected at most {slots} positions, got {pos.numel()}")
        if pos.numel() < slots:
            # Fixed slots: a caller driving fewer active rows than slots leaves
            # the rest inactive rather than having to spell out the -1s.
            pos = torch.cat([pos, torch.full((slots - pos.numel(),), -1, dtype=torch.int32)])
        # The paged cache and page table only represent [0, max_seq_len). A
        # decode step past that indexes off the end of the page table, and the
        # resulting out-of-range paged_update_cache wedges the device rather
        # than failing: it must be rejected on the host.
        limit = self.max_seq_len
        bad = [(slot, int(v)) for slot, v in enumerate(pos.tolist()) if not (-1 <= int(v) < limit)]
        if bad:
            raise ValueError(
                f"decode positions must be in [-1, {limit}) (-1 marks an inactive slot); "
                f"out-of-range slots: {bad[:8]}"
            )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self._pos_dev
        )
        self._host_positions = [int(v) for v in pos.tolist()]
        self.counters["position_refreshes"] += 1

    def set_decode_tokens(self, tokens):
        """Write the decode token input from host (request boundary, teacher
        forcing, or host-sampling mode only)."""
        import torch

        toks = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)
        buf = torch.zeros(SAMPLER_ROWS, dtype=torch.int32)
        buf[: toks.numel()] = toks
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(buf.reshape(1, 1, 1, SAMPLER_ROWS), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self._tokens_dev,
        )
        self.counters["token_input_refreshes"] += 1

    def read_decode_tokens(self, batch=None):
        """Read the sampled tokens back (the one caller-visible readback)."""
        batch = batch or self.max_batch_size
        self.counters["token_readbacks"] += 1
        out = ttnn.to_torch(self._tokens_dev).reshape(-1)[:batch]
        return [int(v) for v in out.tolist()]

    # ------------------------------------------------------------------ decode graph

    def _decode_logits_device(self, *, kv_cache=None, page_table=None, advance_positions=True):
        """One eager device decode over the persistent trace inputs."""
        if advance_positions:
            self._advance_host_positions()
        return self.model.ttnn_decode_forward(
            self._tokens_dev,
            self._pos_dev,
            page_table if page_table is not None else self._page_table_dev,
            kv_cache if kv_cache is not None else self._bound_cache,
            advance_positions=advance_positions,
        )

    def capture_decode_trace(self, *, kv_cache=None, warm=True, warm_at=0):
        """Warm, capture the model decode trace, then capture the sampling trace.

        Runs entirely on dummy state (token 0 at position ``warm_at`` of a
        fresh cache) so the program cache, the sampler pre-compile and both
        captures all happen before any request touches the model. ``reset()``
        afterwards clears the row the warm pass wrote.

        ``warm_at`` exists for :meth:`recapture_decode_traces`, which runs
        mid-request: the warm pass writes one cache row, and the only row it
        may write then is the one the next decode step is about to overwrite
        anyway.
        """
        if self._decode_trace_id is not None:
            return
        cache = kv_cache if kv_cache is not None else self._bound_cache
        if cache is None:
            raise RuntimeError("bind_decode_state must run before capture_decode_trace")

        # Before capture, so the cache-reset zero source is not a post-capture
        # allocation a replay could overwrite (see prepare_cache_reset).
        self.model.prepare_cache_reset(cache)

        self.set_decode_tokens([0] * self.max_batch_size)
        self.set_decode_positions([int(warm_at)] * self.max_batch_size)

        if warm:
            warm_logits = self._decode_logits_device(kv_cache=cache)
            ttnn.synchronize_device(self.mesh_device)
            self.counters["device_synchronizations"] += 1
            if self.sampling is not None:
                # Pre-compile the sampling pipeline while no trace is live: it
                # allocates device buffers a live trace could corrupt on replay.
                self.sampling.precompile(warm_logits, tt_out_tok=self._tokens_dev)
            ttnn.deallocate(warm_logits)

        # The warm pass advanced the device positions and consumed the token
        # buffer; restore the exact capture-time state before recording.
        self.set_decode_tokens([0] * self.max_batch_size)
        self.set_decode_positions([int(warm_at)] * self.max_batch_size)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_logits = self._decode_logits_device(kv_cache=cache)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._decode_trace_id = trace_id

        if self.sampling is not None and not self.host_sampling:
            self.sampling.capture_trace(logits=self._decode_logits, tt_out_tok=self._tokens_dev, skip_precompile=True)
            self._sampling_traced = True

        # Baseline for _maybe_recapture_after_compile: anything cached after
        # this point was compiled while the traces were live.
        self._program_cache_entries_at_capture = self.mesh_device.num_program_cache_entries()

    def recapture_decode_traces(self, *, warm_at=0):
        """Release and re-capture both decode traces.

        Needed after a program is compiled *while a trace is live*. Metal
        registers a trace as active from ``end_mesh_trace`` until it is
        released and flags every allocation made in that window as unsafe
        (``mesh_device.cpp``, ``trace_allocation_tracker.cpp``), because a
        post-capture buffer can land on an address the trace's own freed
        intermediates used and a replay then writes over it. A newly cached
        program keeps a device buffer for the process lifetime, so compiling
        one after capture leaves a permanently unsafe buffer. Re-capturing
        after the compile puts the trace intermediates back on the safe side
        of the program buffers instead of arguing about which addresses
        collide.

        Cheap: 0.3 s on a warm cache, against the >1 s the compile that
        triggered it already cost. Verified with
        ``TT_METAL_TRACE_ALLOC_TRACKING=1`` in ``probe/trace_alloc_probe.py``.
        """
        saved_tokens = self.read_decode_tokens(self.max_batch_size)
        saved_positions = list(self._host_positions) if self._host_positions is not None else None
        if self._decode_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._decode_trace_id)
            self._decode_trace_id = None
        if self._decode_logits is not None:
            ttnn.deallocate(self._decode_logits)
            self._decode_logits = None
        if self.sampling is not None:
            self.sampling.reset_trace()
        self._sampling_traced = False
        self._program_cache_entries_at_capture = None
        self.capture_decode_trace(warm_at=warm_at)
        self.set_decode_tokens(saved_tokens)
        if saved_positions is not None:
            self.set_decode_positions(saved_positions)
        self.counters["trace_recaptures"] += 1

    def _maybe_recapture_after_compile(self, warm_at=0):
        """Re-capture if the program cache grew while the traces were live.

        The trigger is exact rather than heuristic: ``num_program_cache_entries``
        is compared against its value at capture time. On the shipped path a
        single-chunk prompt never trips it, because ``warmup_prefill`` compiles
        every bucket shape *before* capture; a first-use multi-chunk prompt
        does, because its chunk-offset-dependent programs cannot be enumerated
        cheaply (99 offsets at the full context).
        """
        if self._decode_trace_id is None or self._program_cache_entries_at_capture is None:
            return False
        now = self.mesh_device.num_program_cache_entries()
        if now <= self._program_cache_entries_at_capture:
            return False
        self.recapture_decode_traces(warm_at=warm_at)
        return True

    def warmup_prefill(self, lengths=None):
        """Compile the prefill programs for each prefill bucket up front.

        Compiling one 47-layer prefill shape costs ~13 s on this chip, so a
        first request at an unseen shape would pay that inside its TTFT. The
        model buckets prefill lengths (see ``prefill_physical_len``), which
        bounds the distinct shapes; this warms all of them at construction so
        setup work stays out of the measured path.
        """
        self._ensure_owned_state()
        lengths = list(lengths if lengths is not None else self.model.prefill_buckets)
        pad = self.model.pad_token_id
        for n in lengths:
            n = min(int(n), self.max_seq_len)
            if n < 1:
                continue
            logits, _ = self.model.prefill_forward_last_logits_device(
                [pad] * n, kv_cache=self._kv_cache, page_table=self._page_table_dev, user_id=0, seq_len=n
            )
            if self.sampling is not None:
                self.sampling.sample(logits=logits, tt_out_tok=self._prefill_tokens_dev, enable_trace=False)
            ttnn.deallocate(logits)
        # also compile the host-logits (all-positions) terminal path
        self.model.prefill_forward(
            [pad] * min(64, self.max_seq_len),
            kv_cache=self._kv_cache,
            page_table=self._page_table_dev,
            user_id=0,
            seq_len=min(64, self.max_seq_len),
            return_all_logits=True,
        )
        self.reset()

    def _advance_host_positions(self):
        """Mirror the trace's on-device ``plus_one``, and refuse a step that
        would leave the context.

        The captured graph increments the position itself, so nothing on the
        host would otherwise notice a fixed-step loop running off the end of
        the page table - and that does not raise on device, it wedges it
        (work log FM-013). ``-1`` inactive rows are skipped, exactly as
        ``skip_negative_entries=True`` does.
        """
        if self._host_positions is None:
            raise RuntimeError("set_decode_positions must run before a traced decode step")
        limit = self.max_seq_len
        bad = [(slot, p) for slot, p in enumerate(self._host_positions) if p >= 0 and p >= limit]
        if bad:
            raise ValueError(
                f"traced decode would step past the supported context: positions must stay below {limit} "
                f"(the paged cache and page table only represent [0, {limit})); out-of-range slots: {bad[:8]}. "
                "Reset the request or bind a larger max_seq_len."
            )
        self._host_positions = [p if p < 0 else p + 1 for p in self._host_positions]

    def replay_decode_trace(self):
        """Replay the model decode trace only (no sampling), position-checked.

        Use this rather than calling ``ttnn.execute_trace`` directly: the
        captured graph advances the device position itself, so a raw replay
        leaves the host mirror behind and the context guard blind.
        """
        if self._decode_trace_id is None:
            raise RuntimeError("capture_decode_trace must run first")
        self._advance_host_positions()
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=False)
        self.counters["model_trace_replays"] += 1

    def _ensure_sampling_trace(self):
        """Capture the sampling trace if it is missing but usable.

        ``capture_decode_trace`` skips it when the generator is in
        host-sampling mode at capture time, and nothing captured it later, so
        a generator built with ``host_sampling=True`` and then switched back to
        on-device sampling ran the sampler *untraced* for the rest of the
        process: correct tokens, silently slower, no error. Capturing here
        closes that.

        It goes through :meth:`recapture_decode_traces` rather than capturing
        the sampling trace alone, because ``SamplingGenerator.precompile``
        allocates device buffers and doing that while the model trace is live
        is exactly the unsafe-allocation hazard the recapture exists to avoid.
        """
        if self._sampling_traced or self.sampling is None or self.host_sampling:
            return
        if self._decode_trace_id is None:
            return
        active = [p for p in (self._host_positions or []) if p >= 0]
        self.recapture_decode_traces(warm_at=max(active) if active else 0)

    def decode_step_traced(self):
        """One traced token-out decode step. Returns nothing; the sampled token
        is already in the persistent decode token tensor."""
        self._ensure_sampling_trace()
        self.replay_decode_trace()
        if self.host_sampling or self.sampling is None:
            self._host_sample_into_tokens(self._decode_logits)
            return
        self.sampling.sample(
            logits=self._decode_logits, tt_out_tok=self._tokens_dev, enable_trace=self._sampling_traced
        )
        self.counters["sampling_trace_replays"] += 1

    def _host_sample_into_tokens(self, logits):
        """Explicit host-sampling compatibility mode (never the measured path)."""
        import torch

        host = ttnn.to_torch(logits).to(torch.float32)
        self.counters["full_logits_readbacks"] += 1
        self.counters["host_argmax_calls"] += 1
        toks = torch.argmax(host[0, 0, : self.max_batch_size, : self.model.vocab_size], dim=-1)
        self.set_decode_tokens(toks)

    # ------------------------------------------------------------------ low-level contract API

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ):
        """Low-level prefill over caller-owned cache/page table.

        ``tokens``: ``[batch, padded_prompt_len]`` torch int tensor (the caller
        pads; the real per-user length comes from ``prompt_lens``).
        Returns ``[batch, 1, vocab]``, or ``[batch, max(prompt_lens), vocab]``
        when ``return_all_logits`` (shorter users zero-padded on the seq axis).
        """
        import torch

        toks = torch.as_tensor(tokens)
        if toks.dim() == 1:
            toks = toks.unsqueeze(0)
        batch = toks.shape[0]
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens has {len(prompt_lens)} entries for a batch of {batch}")
        pt_dev = page_table if isinstance(page_table, ttnn.Tensor) else self.model.page_table_to_device(page_table)
        outs = []
        for user in range(batch):
            plen = int(prompt_lens[user])
            logits = self.model.prefill_forward(
                toks[user, :plen],
                kv_cache=kv_cache,
                page_table=pt_dev,
                user_id=user,
                seq_len=plen,
                return_all_logits=return_all_logits,
            )
            self.counters["full_logits_readbacks"] += 1
            outs.append(logits)
        if not isinstance(page_table, ttnn.Tensor):
            ttnn.deallocate(pt_dev)
        self._maybe_recapture_after_compile(warm_at=max(int(p) for p in prompt_lens))
        if return_all_logits:
            width = max(o.shape[1] for o in outs)
            padded = [
                torch.nn.functional.pad(o, (0, 0, 0, width - o.shape[1])) if o.shape[1] < width else o for o in outs
            ]
            return torch.cat(padded, dim=0)
        return torch.cat(outs, dim=0)

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        enable_trace: bool = False,
        return_logits: bool = True,
        **kwargs: Any,
    ):
        """Low-level single decode step over caller-owned cache/page table.

        ``tokens``: ``[batch, 1]``; ``start_pos``: ``[batch]``. Returns logits
        ``[batch, vocab]`` (``return_logits``) or the sampled tokens
        ``[batch]``. ``enable_trace`` replays the captured traced token-out
        path (the cache/page table must be the bound ones).
        """
        import torch

        toks = torch.as_tensor(tokens).reshape(-1)
        pos = torch.as_tensor(start_pos).reshape(-1)
        if self._tokens_dev is None:
            self.bind_decode_state(kv_cache=kv_cache, page_table=page_table)
        if enable_trace and self._bound_cache is not None and kv_cache is not self._bound_cache:
            raise ValueError(
                "traced decode replays a graph bound to the cache given to bind_decode_state / "
                "capture_decode_trace; pass that cache, or rebind and recapture before switching."
            )
        if isinstance(page_table, ttnn.Tensor):
            if enable_trace and page_table is not self._page_table_dev:
                raise ValueError(
                    "traced decode replays a graph bound to the page-table tensor given to "
                    "bind_decode_state / capture_decode_trace; pass that tensor, or hand a torch page "
                    "table so it can be copied into the bound buffer."
                )
        else:
            if self._page_table_caller_owned:
                raise ValueError(
                    "the bound page table is a caller-owned device tensor; update it in place instead of "
                    "passing a torch page table, or rebind with a torch table so the generator owns the copy."
                )
            # Generator-owned buffer: only copy when the table actually changed.
            self.refresh_page_table(page_table, only_if_changed=True)
        self.set_decode_tokens(toks)
        self.set_decode_positions(pos)

        if enable_trace:
            if self._decode_trace_id is None:
                self.capture_decode_trace(kv_cache=kv_cache)
                self.set_decode_tokens(toks)
                self.set_decode_positions(pos)
            self.decode_step_traced()
            if return_logits:
                out = ttnn.to_torch(self._decode_logits).to(torch.float32)
                self.counters["full_logits_readbacks"] += 1
                return out[0, 0, : toks.numel(), : self.model.vocab_size]
            return torch.tensor(self.read_decode_tokens(toks.numel()), dtype=torch.int32)

        logits = self._decode_logits_device(
            kv_cache=kv_cache, page_table=page_table if isinstance(page_table, ttnn.Tensor) else None
        )
        self.counters["eager_decode_steps"] += 1
        out = ttnn.to_torch(logits).to(torch.float32)
        self.counters["full_logits_readbacks"] += 1
        ttnn.deallocate(logits)
        if return_logits:
            return out[0, 0, : toks.numel(), : self.model.vocab_size]
        self.counters["host_argmax_calls"] += 1
        return torch.argmax(out[0, 0, : toks.numel(), : self.model.vocab_size], dim=-1)

    # ------------------------------------------------------------------ high-level contract API

    def prefill_logits(self, prompt_token_ids: List[int]):
        """Prefill one prompt and return all logits ``[1, prompt_len, vocab]``."""
        self._ensure_owned_state()
        self.reset()
        seq = len(prompt_token_ids)
        try:
            return self.model.prefill_forward(
                prompt_token_ids,
                kv_cache=self._kv_cache,
                page_table=self._page_table_dev,
                user_id=0,
                seq_len=seq,
                return_all_logits=True,
            )
        finally:
            # The all-positions host path compiles its own tile-aligned slabs,
            # so it can leave program buffers on the unsafe side of a live
            # trace even though it never replays one itself (FM-016).
            self._maybe_recapture_after_compile(warm_at=seq)

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input=None,
        enable_trace: bool = True,
        stop_on_eos: bool = True,
        host_sampling: Optional[bool] = None,
        sampling_params: Optional[SamplingParams] = None,
        return_timing: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """Greedy (or ``sampling_params``-driven) generation for a single user.

        Returns the model's own predictions, one per requested token, even when
        ``next_input`` teacher-forces a different next input. ``enable_trace``
        is honoured explicitly: ``True`` runs every decode step through the
        captured model + sampling traces.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        seq = len(prompt_token_ids)
        if seq + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt {seq} + {max_new_tokens} new tokens exceeds the supported context {self.max_seq_len}"
            )
        prev_host_sampling = self.host_sampling
        if host_sampling is not None:
            self.host_sampling = host_sampling
        if sampling_params is not None:
            self.set_sampling_params(sampling_params)

        try:
            self._ensure_owned_state()
            if enable_trace and self._decode_trace_id is None:
                self.capture_decode_trace()
                self.reset()
            t_reset = time.perf_counter()
            self.reset()
            t_start = time.perf_counter()
            # ---- prefill + first token (sampled on device from the prefill logits) ----
            first = self._prefill_and_sample_first(prompt_token_ids, recapture=False)
            t_first = time.perf_counter()
            # A first-use prefill shape compiles programs while the traces are
            # live (the terminal slice/pad depends on seq mod 32, and a
            # multi-chunk prompt adds chunk-offset programs); re-capture before
            # any replay so those program buffers are not on the unsafe side of
            # the traces. No-op on a shape warmed at construction. Timed
            # separately: it is a one-time setup cost for that shape and
            # belongs in neither TTFT nor the decode rate.
            if enable_trace:
                self._maybe_recapture_after_compile(warm_at=seq)
            t_ready = time.perf_counter()
            predictions = [first]
            self.set_decode_positions([seq] + [0] * (self.max_batch_size - 1))
            if next_input is not None:
                forced = int(next_input(0, first))
                self.set_decode_tokens([forced] + [0] * (self.max_batch_size - 1))
            elif stop_on_eos and first in self.model.eos_token_ids:
                elapsed = {
                    "reset_s": t_start - t_reset,
                    "ttft_s": t_first - t_start,
                    "recapture_s": t_ready - t_first,
                    "decode_s": 0.0,
                    "decode_tokens": 0,
                }
                return (predictions, elapsed) if return_timing else predictions

            # ---- decode loop ----
            for step in range(1, max_new_tokens):
                if enable_trace:
                    self.decode_step_traced()
                else:
                    logits = self._decode_logits_device()
                    self.counters["eager_decode_steps"] += 1
                    if self.host_sampling or self.sampling is None:
                        self._host_sample_into_tokens(logits)
                    else:
                        self.sampling.sample(logits=logits, tt_out_tok=self._tokens_dev, enable_trace=False)
                    ttnn.deallocate(logits)
                token = self.read_decode_tokens(1)[0]
                predictions.append(token)
                if next_input is not None:
                    forced = int(next_input(step, token))
                    self.set_decode_tokens([forced] + [0] * (self.max_batch_size - 1))
                elif stop_on_eos and token in self.model.eos_token_ids:
                    break
            t_end = time.perf_counter()
        finally:
            self.host_sampling = prev_host_sampling

        if return_timing:
            timing = {
                # The request-boundary cache reset is reported separately, not
                # folded into TTFT: it is drained before the clock starts.
                "reset_s": t_start - t_reset,
                "ttft_s": t_first - t_start,
                # Zero unless this prompt shape compiled programs after capture.
                "recapture_s": t_ready - t_first,
                "decode_s": t_end - t_ready,
                "decode_tokens": len(predictions) - 1,
            }
            return predictions, timing
        return predictions

    def _prefill_and_sample_first(self, prompt_token_ids, user_id: int = 0, *, recapture: bool = True):
        """Prefill and pick the first generated token with the same on-device
        sampler the decode loop uses.

        The final prompt position lands at row ``row`` of the 32-row sampler
        tile, so the sampled tokens go to a scratch output and the wanted one
        is placed in decode slot 0 (a one-word request-boundary refresh, not a
        per-token host step).

        ``recapture=False`` is for :meth:`generate`, which runs the same
        post-compile recapture itself so it can time it apart from TTFT. Every
        other caller wants the default: a first-use prefill shape compiles
        programs while the traces are live, and replaying over those program
        buffers is the unsafe-allocation hazard in work log FM-016.
        """
        import torch

        seq = len(prompt_token_ids)
        logits_dev, row = self.model.prefill_forward_last_logits_device(
            prompt_token_ids,
            kv_cache=self._kv_cache,
            page_table=self._page_table_dev,
            user_id=user_id,
            seq_len=seq,
        )
        if self.host_sampling or self.sampling is None:
            host = ttnn.to_torch(logits_dev).to(torch.float32)
            self.counters["full_logits_readbacks"] += 1
            self.counters["host_argmax_calls"] += 1
            token = int(torch.argmax(host[0, 0, row, : self.model.vocab_size]).item())
            ttnn.deallocate(logits_dev)
        else:
            # Untraced: the prefill logits are a fresh allocation every request.
            self.sampling.sample(logits=logits_dev, tt_out_tok=self._prefill_tokens_dev, enable_trace=False)
            ttnn.deallocate(logits_dev)
            self.counters["token_readbacks"] += 1
            token = int(ttnn.to_torch(self._prefill_tokens_dev).reshape(-1)[row].item())
        toks = [0] * self.max_batch_size
        toks[user_id] = token
        self.set_decode_tokens(toks)
        if recapture:
            self._maybe_recapture_after_compile(warm_at=len(prompt_token_ids))
        return token

    def reset(self) -> None:
        """Wipe per-prompt state. Keeps weights, device buffers and traces.

        Drains before returning. Zeroing the full-context cache enqueues 47
        device-to-device copies of a 118 MiB buffer, and leaving them in flight
        made every later measurement in the same call ambiguous: the request's
        TTFT absorbed the drain of work that belongs to the request boundary
        (work log FM-016). One synchronization per request is cheap and makes
        the contract ("the cache is zeroed") true on return.
        """
        self._ensure_owned_state()
        self.model.reset_kv_cache(self._kv_cache if self._kv_cache is not None else self._bound_cache)
        ttnn.synchronize_device(self.mesh_device)
        self.counters["device_synchronizations"] += 1
        self.counters["kv_cache_resets"] += 1
        self.set_decode_tokens([0] * self.max_batch_size)
        self.set_decode_positions([0] * self.max_batch_size)
        if self.sampling is not None:
            self.sampling.reset_penalty_counts()

    def reset_counters(self):
        self.counters = _new_counters()

    def teardown(self):
        if self._decode_trace_id is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._decode_trace_id)
            except Exception:
                pass
            self._decode_trace_id = None
        if self.sampling is not None:
            try:
                self.sampling.reset_trace()
            except Exception:
                pass
            self._sampling_traced = False


# --------------------------------------------------------------------- factory


def build_generator(model_dir, mesh_device, **kwargs) -> GLM47FlashGenerator:
    """Standard Metal readiness factory.

    ``model_dir`` is the autoport directory (``models/autoports/zai_org_glm_4_7_flash``);
    the checkpoint itself is resolved from ``checkpoint_dir=``, the
    ``GLM47_FLASH_SNAPSHOT`` env var, or the local HF cache.
    """
    model_dir = Path(model_dir)
    hf_model_id = kwargs.pop("hf_model_id", DEFAULT_HF_MODEL_ID)
    checkpoint_dir = kwargs.pop("checkpoint_dir", None)
    host_sampling = bool(kwargs.pop("host_sampling", False))
    enable_sampling = bool(kwargs.pop("enable_sampling", True))
    capture_trace = bool(kwargs.pop("capture_trace", True))
    warmup_prefill_lens = kwargs.pop("warmup_prefill_lens", "buckets")
    tokenizer = kwargs.pop("tokenizer", None)
    progress = kwargs.pop("progress", print)

    snapshot = resolve_checkpoint_dir(checkpoint_dir, hf_model_id)
    model = GLM47FlashModel.from_pretrained(
        mesh_device,
        checkpoint_dir=snapshot,
        hf_model_id=hf_model_id,
        progress=progress,
        **kwargs,
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)

    generator = GLM47FlashGenerator(
        model, tokenizer=tokenizer, host_sampling=host_sampling, enable_sampling=enable_sampling
    )
    generator._ensure_owned_state()
    if warmup_prefill_lens:
        lens = None if warmup_prefill_lens == "buckets" else warmup_prefill_lens
        t0 = time.perf_counter()
        generator.warmup_prefill(lens)
        progress(f"  prefill programs warmed in {time.perf_counter() - t0:.1f}s")
    if capture_trace:
        t0 = time.perf_counter()
        generator.capture_decode_trace()
        generator.reset()
        progress(f"  decode + sampling traces captured in {time.perf_counter() - t0:.1f}s")
    return generator
