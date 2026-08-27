# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM serving adapter for Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies.

This file is **translation only**. Every device operation it causes is a call
into ``tt/generator.py``'s low-level surface -- ``prefill_forward`` /
``decode_forward`` / ``set_sampling_params`` / ``configure_paging`` -- which is
the same surface the standalone readiness runners drive. There is no model code
here, no second sampler, no host argmax, no full-logits readback on the measured
path and no Python readback/writeback token-feedback loop.

The three things it actually has to reconcile
---------------------------------------------

**1. Who owns the cache.** Standalone, the generator allocates its own paged
cache and builds its own page tables. Serving, vLLM owns both: it picks the
block size and the block count, calls ``allocate_kv_cache`` once, and hands that
exact cache and a ``[batch, max_num_blocks_per_req]`` block table into every
forward. ``allocate_kv_cache`` therefore installs vLLM's geometry through
``Qwen3CoderGenerator.configure_paging`` and allocates the cache through
``Qwen3CoderModel.allocate_kv_cache(num_blocks=...)``; the generator's own
``_kv_cache`` is never created, so no standalone-cache assumption can survive.

**2. Who owns the token.** The generator's traced decode path writes the sampled
token straight back into the persistent decode token input with ``tt_out_tok``
and advances ``current_pos``/``rotary_position`` on device with
``ttnn.plus_one``. So after step *N* the **device**, not the host, holds the
token and position step *N+1* needs. vLLM's scheduler also tracks them, and
under ``--async-scheduling`` its copy is a step behind. On a steady decode step
this adapter therefore passes *nothing* -- ``decode_forward(None, None,
page_table=..., ...)`` replays the two traces over state the device already
owns. Only when vLLM says the slot layout changed (``reset_batch``, or the step
right after a prefill) does it reinstall host state, and even then it keeps the
device's token and position for slots that are merely continuing
(``_merge_scheduler_view``), so an async-ahead scheduler cannot re-decode a
position or feed back a stale token.

**3. Who owns sampling.** vLLM would rather hand us logits. It only stops doing
that when the model declares ``supports_sample_on_device`` and the server runs
with ``sample_on_device_mode: all`` -- then it sends per-row
``(temperature, top_k, top_p)`` and expects token ids back. That is the measured
path and it is exactly the full model's canonical split sampling: greedy takes
``Qwen3CoderModel.sample_greedy_argmax``, anything sampled takes
``sample_split``, both are ``_WatcherCleanSampling1D``, both are traced, both
write ``tt_out_tok``. The TT plugin still routes a few request shapes to host
sampling on its own (logprobs on a mesh that is not 8 or 32 dies, ``min_p``,
``bad_words``, ``logit_bias``, structured output) -- for those it passes
``sampling_params=None`` and wants logits. That is served by the generator's
pre-existing, explicit ``sampling_mode="host"`` compatibility mode. It is opt-in
per request by vLLM, never used for a performance number, and it never displaces
the traced path. The eager decode does release the captured decode traces (it
allocates, so ``generator.decode_forward`` calls
``_release_decode_traces_before_allocating``); the adapter therefore sets
``_needs_decode_install`` and the next device-sampled step re-captures through
``_refresh_trace_state`` rather than replaying a stale trace.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator import SAMPLING_SLOTS, Qwen3CoderGenerator, build_generator
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.model import HF_MODEL_ID, MAX_CONTEXT

#: This port's directory. Everything below reads its policy from here, not from
#: a vLLM flag, so serving cannot silently run a different model than readiness.
MODEL_DIR = Path(__file__).resolve().parents[1]

#: The datatype-sweep selection (stage 07). ``build_generator`` accepts a path
#: and threads it into ``Qwen3CoderModel``, so weight groups, activation dtype,
#: CCL dtype, KV-cache dtype, compute fidelities and layer exceptions all come
#: from this one file on the serving path exactly as they do on the readiness
#: path. ``QWEN3_PRECISION_CONFIG`` still overrides it, for sweeps.
SELECTED_PRECISION_CONFIG = MODEL_DIR / "config" / "selected_precision_config.json"

#: ``config/context_contract.json`` is the single source of truth for served
#: context. ``get_max_tokens_all_users`` and ``initialize_vllm_model`` both read
#: it rather than trusting a CLI value, so a ``--max-model-len`` above the
#: recorded capability fails loudly instead of serving a quietly-clipped model.
CONTEXT_CONTRACT = MODEL_DIR / "config" / "context_contract.json"

#: Whether a serving prefill may run while the decode traces stay captured.
#: **Off by default, and that is a measurement, not caution.** Keeping them alive
#: hangs the NoC: prefill's collectives share the sampler's persistent CCL
#: buffers and semaphores with the captured graph, and after a few admissions a
#: replay waits on a semaphore value an eager collective already consumed.
#: ``doc/vllm_integration/triage/tt-triage-preserve-traces-hang.txt`` is that
#: hang, caught with ``dump_running_operations`` reporting ``NOC0 CB0..3 active
#: (0xFFFFFFFF). NoC is likely hung.`` on device 0. Releasing on prefill --
#: which is what ``Qwen3CoderGenerator.prefill_forward`` has always done -- makes
#: the next capture re-establish that state, and
#: ``Qwen3CoderGenerator._decode_compiled_keys`` keeps the re-capture from
#: paying for a second eager warm pass. Set ``QWEN3_VLLM_PRESERVE_DECODE_TRACES=1``
#: to reproduce the hang.
PRESERVE_DECODE_TRACES = os.getenv("QWEN3_VLLM_PRESERVE_DECODE_TRACES", "0") not in ("0", "", "false", "no")


#: Prefix caching is ON by default from phase 3. This is no longer a feature gate
#: but a kill switch: ``QWEN3_PREFIX_CACHING=0`` restores the pre-phase-3 refusal
#: without touching ``model_capabilities``, so the two can be reverted separately.
_PREFIX_CACHING_ENABLED = os.getenv("QWEN3_PREFIX_CACHING", "1") not in ("0", "", "false", "no")


def _supported_context() -> int:
    try:
        contract = json.loads(CONTEXT_CONTRACT.read_text())
    except (OSError, ValueError):
        return MAX_CONTEXT
    return int(contract.get("current_supported_context") or MAX_CONTEXT)


def _reorder_history(history, order, picks):
    """Reorder a vLLM ``[rows, L]`` token history into graph-row order.

    Type-preserving **on purpose**. ``Generator._row_token_ids`` consumes these
    with ``torch.as_tensor(history)``, which raises ``TypeError: only integer
    tensors of a single element can be converted to an index`` when handed a
    *list of 1-D tensors*. Rebuilding a tensor history with a list comprehension
    therefore turns a working penalised decode into a crash -- and only when the
    width ladder is active, since with the ladder off ``order`` is ``None`` and
    the history is passed through untouched. That is exactly the shape of bug
    that reaches production: invisible on the default path, fatal on the new one.

    Anything that supports fancy indexing (torch tensors, numpy arrays) is
    indexed so its type survives; genuine python sequences keep the list
    comprehension, which is already what ``_row_token_ids`` expects from them.
    """
    if history is None:
        return None
    if isinstance(history, torch.Tensor):
        return history[order.to(torch.long)]
    if hasattr(history, "__getitem__") and hasattr(history, "dtype"):  # numpy & friends
        return history[list(picks)]
    return [history[i] for i in picks]


def _as_int_list(values, length: int, default) -> list:
    """vLLM hands per-row sampling params over as python lists; normalise them."""
    if values is None:
        return [default] * length
    # vLLM hands prompt_lens/start_pos over as numpy arrays and sampling params
    # as python lists, and the plugin builds some fields from torch tensors.
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values] * length
    out = list(values)[:length]
    out.extend([default] * (length - len(out)))
    return out


class Qwen3CoderForCausalLM:
    """The class vLLM instantiates. Registered as ``TTQwen3MoeForCausalLM``.

    ``Qwen3MoeForCausalLM`` is the architecture string in this checkpoint's
    ``config.json``; the TT plugin registers every model under a ``TT`` prefix.
    """

    #: Read off the *class* by ``vllm_tt_plugin.platform.check_and_update_config``
    #: before anything is instantiated.
    #:
    #: * ``supports_sample_on_device`` -- the full model's traced split sampling
    #:   is the measured token-out path; without this flag the readiness runner's
    #:   ``sample_on_device_mode: all`` is a hard config error.
    #: * ``supports_async_decode`` -- ``decode_forward(read_from_device=False)``
    #:   returns device handles, ``read_decode_output(async_read=True)`` does the
    #:   deferred read and records an event, ``process_decode_output_host`` does
    #:   host formatting only. This also gates ``--async-scheduling``, which is
    #:   safe here because ``_merge_scheduler_view`` prefers the device's token
    #:   and position over an async-ahead scheduler's.
    #: * ``supports_prefix_caching`` -- **False**. Phase 3 REVERTED it: with caching
    #:   on, cold-vs-warm greedy output matched on only 1 of 10 prompts, while the
    #:   same test with --no-enable-prefix-caching matched 10/10. See
    #:   doc/prefix_caching/probes/phase3_cold_warm_rate.json and
    #:   phase3_control_no_prefix_caching.json. The adapter wiring below is correct
    #:   and stays; the flag must not go back to True until that gap is closed.
    #:   (Historical note, kept because the wiring depends on it:)
    #:   vLLM then sends a non-zero ``start_pos`` (= ``num_computed_tokens``) with
    #:   the FULL prompt and the FULL ``prompt_lens``; the model slices the suffix
    #:   itself, matching tt_transformers' ``tokens[i, num_cached:seq_len]``.
    #:   Two vLLM invariants make our generator-side guards exact rather than
    #:   defensive, both READ OFF vllm rather than assumed:
    #:     - ``max_cache_hit_length = request.num_tokens - 1`` (kv_cache_manager.py)
    #:       so a full hit still recomputes the last token: ``start < prompt_len``.
    #:     - cache hits are whole blocks and ``allocate_slots`` requires
    #:       block-aligned ``num_computed_tokens``: ``start % 32 == 0``.
    #:   Chunked prefill is force-disabled by the plugin platform, so prefix
    #:   caching is the only source of a non-zero ``start_pos``.
    #:   Kill switch: ``QWEN3_PREFIX_CACHING=0`` restores the old refusal.
    model_capabilities = {
        # QUALITY-GATE EDIT (doc/prefix_caching/QUALITY_BAR.md): flipped True to
        # run the caching-ON arm. Revert this single line to False if the gate
        # fails. See doc/prefix_caching/quality_gate/.
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    # -- construction ---------------------------------------------------------

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len: int | None = None,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "Qwen3CoderForCausalLM":
        if int(tt_data_parallel) != 1:
            raise ValueError(
                f"tt_data_parallel={tt_data_parallel} is unsupported: this port occupies the whole "
                "1x4 mesh with tensor parallelism, so there is no submesh left to replicate onto."
            )
        supported = _supported_context()
        max_seq_len = supported if max_seq_len is None else int(max_seq_len)
        if max_seq_len > supported:
            raise ValueError(
                f"--max-model-len {max_seq_len} exceeds the context recorded in {CONTEXT_CONTRACT} " f"({supported})."
            )
        max_batch_size = int(max_batch_size)
        if not 1 <= max_batch_size <= SAMPLING_SLOTS:
            raise ValueError(
                f"max_num_seqs={max_batch_size} is outside [1,{SAMPLING_SLOTS}]. "
                "nlp_create_qkv_heads_decode and ttnn.sampling both address 32 fixed user slots."
            )

        # Reduced serving target for the bring-up inner loop only: the same
        # adapter, generator, registration, cache/page-table shapes, terminal
        # norm/LM head, sampler and trace behaviour, with fewer copies of the one
        # layer kind this model has. Never used for accuracy or performance
        # evidence -- the final run leaves it unset and gets all 48 layers.
        reduced = os.getenv("QWEN3_VLLM_NUM_LAYERS")
        if reduced:
            n_layers = int(reduced)
            logger.warning(
                "QWEN3_VLLM_NUM_LAYERS={} -- REDUCED serving target, bring-up inner loop only. "
                "Do not report accuracy or performance from this server.",
                n_layers,
            )

        precision = os.getenv("QWEN3_PRECISION_CONFIG") or str(SELECTED_PRECISION_CONFIG)
        generator = build_generator(
            MODEL_DIR,
            mesh_device,
            max_batch_size=max_batch_size,
            max_context_len=max_seq_len,
            # The traced decode loop advances ``rotary_position`` on device and
            # nothing on device clamps it, so the cos/sin tables must already
            # cover every position this server may be asked to serve. Sizing
            # them here means no serving step can ever grow them -- growing
            # reallocates, and a captured trace holds the old identities.
            rope_cache_len=max_seq_len,
            precision=precision,
            **({} if n_layers is None else {"override_num_layers": int(n_layers)}),
        )
        # Logged after the generator exists so ``active_row_gating`` is read off
        # the model that was actually built rather than re-parsed from the
        # environment here. Every leg of an A/B then carries its own
        # configuration in its own server log, instead of the two legs being
        # distinguishable only by the work-log prose that says which was which.
        logger.info(
            "Qwen3-Coder-30B-A3B vLLM init: max_num_seqs={} max_model_len={} precision={} "
            "active_row_gating={} optimizations={}",
            max_batch_size,
            max_seq_len,
            precision,
            generator.model.active_row_gating,
            optimizations,
        )
        return cls(generator, max_model_len=max_seq_len, max_num_seqs=max_batch_size)

    def __init__(self, generator: Qwen3CoderGenerator, *, max_model_len: int, max_num_seqs: int):
        self.generator = generator
        self.model = generator.model
        self.mesh_device = generator.mesh_device
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)

        #: vLLM-owned cache; set by ``allocate_kv_cache`` and never re-created.
        self.kv_cache: list | None = None
        #: True until a decode step has installed host state into the trace.
        #: Every prefill sets it, because prefill admits a new request into a
        #: slot whose device token/position belong to whoever held it before.
        self._needs_decode_install = True
        #: Set by ``warmup_model_prefill`` so the plugin's two-phase warmup does
        #: not repeat the prefill sweep (the plugin resets this itself).
        self.already_warmed_up_prefill = False
        #: Runtime-fallback bookkeeping, reported by ``serving_audit``.
        self._audit = {
            "device_sampled_decode_steps": 0,
            "host_sampled_decode_steps": 0,
            "device_sampled_prefills": 0,
            "host_sampled_prefills": 0,
            "decode_trace_installs": 0,
            "top_k_clamped_requests": 0,
            "penalised_decode_steps": 0,
            "ignored_seed_requests": 0,
            #: Steps on which vLLM actually took the async split -- i.e. called
            #: ``read_decode_output(async_read=True)`` rather than reading the
            #: device handle synchronously inside ``execute_model``. This is what
            #: makes ``supports_async_decode`` a measurement instead of a claim;
            #: see the one-time log line in ``read_decode_output``.
            "async_decode_reads": 0,
            "sync_decode_reads": 0,
        }
        self._warned: set[str] = set()

        #: Decode graph widths this server may capture, ascending. Read from
        #: ``QWEN3_DECODE_WIDTHS`` (comma-separated, e.g. ``1,8,32``); anything
        #: above ``max_num_seqs`` is dropped and ``max_num_seqs`` is always
        #: present, so the default -- unset -- is exactly the shipped single
        #: fixed-width graph and nothing below changes behaviour.
        #:
        #: Why this exists: expert, router and paged-SDPA cost is paid per row
        #: **configured**, not per row live (``doc/optimized_vllm/README.md``'s
        #: control curve: 227.9 ms fixed + 1.28 ms x live_rows at 32 slots). The
        #: only lever that removes the fixed term is a graph with fewer rows.
        self._decode_widths = self._configured_widths()
        #: Graph row -> vLLM slot for the live trace, or ``None`` when the graph
        #: is full width and the mapping is the identity. Rewritten only on an
        #: install, which is the only step on which the batch layout may change
        #: (``model_runner.py`` sets ``reset_batch`` from a sticky
        #: ``_decode_layout_changed_since_last_decode``).
        self._compaction: torch.Tensor | None = None
        #: One entry per decode forward that is still awaiting its host read: the
        #: graph-row -> vLLM-slot mapping **that forward was issued with**.
        #:
        #: Why a queue and not just ``_compaction``. The un-permutation happens in
        #: ``process_decode_output_host``, which under ``--async-scheduling`` does
        #: not run in the same step as the forward that produced the tokens: the
        #: forward returns a device handle and the host read happens later. A
        #: mapping stored on the adapter can therefore be **rewritten by a later
        #: install before the earlier step's tokens are scattered**, which would
        #: put every token on the wrong slot -- silently, as a correctness bug.
        #:
        #: The plugin happens to order this safely today: only a layout change can
        #: rewrite the mapping, and ``model_runner.py`` drains pending async decodes
        #: whenever the layout changed. But that is an invariant of a *different*
        #: repository, which this one must not modify and cannot pin with a test.
        #: Pairing each output with the mapping its own forward used replaces
        #: that dependency with a weaker and detectable one. It no longer relies
        #: on drain-on-layout-change; it does still assume the plugin finalizes
        #: decode steps in issue order and exactly once, which is an invariant of
        #: the same foreign repository. The difference is that a violation now
        #: raises (tag mismatch, underflow, or depth cap) instead of silently
        #: scattering a step's tokens through another step's permutation.
        self._pending_orders: deque = deque()
        #: Monotonic id of the next decode forward to be issued, and of the next
        #: output expected. They are compared on every pop: a queue that has
        #: skipped or reordered an entry shows up as a tag mismatch rather than
        #: as tokens quietly landing on the wrong requests.
        self._next_issue_tag = 0
        self._next_output_tag = 0
        #: Hard ceiling on outstanding decode forwards. Async scheduling runs at
        #: most a step or two ahead; anything approaching this is a leak, not
        #: depth.
        self._pending_orders_cap = 64
        self._audit["narrow_decode_installs"] = 0
        self._audit["decode_graph_width"] = self.max_num_seqs
        #: Times the output path found no queued mapping. Must stay 0; a nonzero
        #: value means forwards and host reads are not paired one-to-one. The
        #: path raises rather than guessing -- applying the adapter's current
        #: mapping here would be the exact mis-scatter the queue exists to
        #: prevent -- so this counter records a raise, not a silent fallback.
        self._audit["compaction_fifo_underflows"] = 0
        self._audit["compaction_fifo_max_depth"] = 0

    @property
    def _compaction_enabled(self) -> bool:
        """Whether the row-mapping queue is in use at all.

        Derived from ``_decode_widths`` rather than cached, because probes and
        tests rebind that list after construction to switch the ladder on and
        off; a cached flag would go stale and silently disable the pairing.

        With the ladder disabled there is no permutation to pair, so nothing is
        pushed or popped and the shipped path gains neither the bookkeeping nor
        its failure modes -- in particular it cannot raise the errors below.
        """
        return len(self._decode_widths) > 1

    def _reset_pending_orders(self) -> None:
        """Drop queued mappings whose outputs can no longer be read.

        Called wherever the decode traces are released or replaced. A queued
        entry refers to a forward whose sampled tokens live in the trace's
        persistent output tensor; once that trace is gone the handle cannot be
        read at all, so the entry is dead and keeping it would desync every
        later pop. The tags are realigned rather than zeroed so the invariant
        ("the n-th output pairs with the n-th forward") survives the reset.
        """
        self._pending_orders.clear()
        self._next_output_tag = self._next_issue_tag

    #: The ladder used when ``QWEN3_DECODE_WIDTHS`` is unset. Powers of two to
    #: ``max_num_seqs``: each step runs in the narrowest graph that holds the
    #: live rows, so the cost a user pays tracks occupancy instead of the slot
    #: count the server was configured with.
    #:
    #: On by default because the alternative default is *known wrong*: a
    #: ``max_num_seqs=32`` server decodes a single user at 4.3464 t/s/u against
    #: 49.3636 with the ladder, and a deployment that simply does not set an
    #: environment variable gets the slow one. That is the same failure shape as
    #: a missing ``sample_on_device_mode`` -- a config key whose absence looks
    #: like broken hardware rather than a default.
    #:
    #: ``QWEN3_DECODE_WIDTHS=32`` (or any single width equal to ``max_num_seqs``)
    #: restores the previous fixed-width behaviour exactly.
    DEFAULT_DECODE_WIDTHS = (1, 2, 4, 8, 16, 32)

    def _configured_widths(self) -> list[int]:
        """Decode graph widths this server may capture, ascending.

        ``max_num_seqs`` is always present -- a graph can never be wider than
        the slots the caller sends, and the full-width graph must exist as the
        fallback -- and anything above it is dropped.
        """
        raw = os.getenv("QWEN3_DECODE_WIDTHS", "").strip()
        source = raw.split(",") if raw else [str(w) for w in self.DEFAULT_DECODE_WIDTHS]
        widths = {self.max_num_seqs}
        for piece in source:
            piece = piece.strip()
            if not piece:
                continue
            value = int(piece)
            if 1 <= value <= self.max_num_seqs:
                widths.add(value)
        return sorted(widths)

    def _choose_width(self, live_rows: int) -> int:
        """Narrowest configured graph that can hold ``live_rows`` requests."""
        for width in self._decode_widths:
            if width >= max(1, live_rows):
                return width
        return self.max_num_seqs

    @staticmethod
    def _compaction_order(host_positions: torch.Tensor, width: int, rows: int) -> torch.Tensor:
        """Graph row -> vLLM slot, live slots first, then spare slots.

        The live slots go to rows ``0..live-1`` in their original order; the
        remaining graph rows are filled from the *inactive* slots so that every
        graph row still names a distinct vLLM slot and therefore still carries a
        real (zero-filled) page-table row. Those rows install ``current_pos =
        -1``, which is the inactive sentinel the traced graph already relies on.

        This is a permutation of **only** the three per-row inputs -- position,
        rotary index and page-table row -- plus the token. No KV page moves: the
        cache is reached exclusively through page-table entries, so a request's
        pages are wherever its page-table row says they are, in whatever graph
        row that row is installed.
        """
        live = torch.nonzero(host_positions >= 0, as_tuple=False).reshape(-1)
        spare = torch.nonzero(host_positions < 0, as_tuple=False).reshape(-1)
        order = torch.cat((live, spare))[:width]
        if order.numel() < width:  # fewer vLLM slots than graph rows: cannot happen
            raise RuntimeError(f"cannot fill a {width}-row graph from {rows} slots")
        return order.to(torch.int64)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    # -- scheduler sizing -----------------------------------------------------

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Total KV tokens vLLM may allocate blocks for, across all users.

        The whole advertised context for one user. ``config/context_contract.json``
        records 262144 as the supported context and the paged decode probe that
        reached position 262143; this is what makes vLLM size enough blocks for
        a single request to actually use it. At this port's 4 dies the KV cost is
        512 B per token per layer per die over 48 layers -- 24 KiB per token per
        die, so 262144 tokens is 6.29 GiB of the 34.18 GiB each die reports.

        The worker adds ``block_size * max_num_seqs`` of its own headroom on top
        and converts to blocks, so nothing here needs to model that.
        """
        supported = _supported_context()
        return supported if max_model_len is None else min(int(max_model_len), supported)

    # -- vLLM-owned KV cache --------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers: int):
        """Allocate the attention KV cache **for vLLM**, at vLLM's geometry.

        ``kv_cache_shape`` is ``(num_blocks, num_kv_heads_per_device, block_size,
        head_dim)``; the plugin has already divided the head count by the mesh
        size, which for this port's 4 dies and 4 KV heads gives the 1 local head
        per die the model expects. The block size is vLLM's, so it is installed
        into the generator here -- before warmup, before any forward, before any
        trace -- rather than assumed.

        ``dtype`` is vLLM's torch dtype and is deliberately **not** used: the KV
        dtype is part of the selected precision policy
        (``kv_cache_dtype`` in ``selected_precision_config.json``) and serving
        must not silently run a different one than the sweep measured.
        """
        num_blocks, kv_heads, block_size, head_dim = (int(v) for v in kv_cache_shape)
        if num_layers != self.model.num_layers:
            if not os.getenv("QWEN3_VLLM_NUM_LAYERS"):
                raise ValueError(f"vLLM asked for {num_layers} attention layers, model has {self.model.num_layers}")
            # Reduced bring-up target: vLLM sized blocks for the real depth, the
            # model only has a few layers. Allocating the model's depth is the
            # right thing -- it is strictly less memory and the page geometry,
            # which is what this loop is testing, is unchanged.
            logger.warning(
                "Reduced target: allocating {} layer caches, vLLM planned for {}",
                self.model.num_layers,
                num_layers,
            )
        if kv_heads != self.model.config.local_attention.num_key_value_heads:
            raise ValueError(
                f"vLLM computed {kv_heads} local KV heads, this port shards to "
                f"{self.model.config.local_attention.num_key_value_heads} per die"
            )
        if head_dim != self.model.head_dim:
            raise ValueError(f"vLLM head_dim {head_dim} != model head_dim {self.model.head_dim}")

        pages_per_user = min(math.ceil(self.max_model_len / block_size), num_blocks)
        self.generator.configure_paging(
            page_block_size=block_size, pages_per_user=pages_per_user, num_blocks=num_blocks
        )
        logger.info(
            "vLLM-owned KV cache: {} blocks x {} tokens = {} tokens, {} local KV heads, dtype {} "
            "(from the selected precision config; vLLM asked for {})",
            num_blocks,
            block_size,
            num_blocks * block_size,
            kv_heads,
            self.model.precision.kv_cache_dtype,
            dtype,
        )
        self.kv_cache = self.model.allocate_kv_cache(num_blocks=num_blocks)
        return self.kv_cache

    # -- warmup ---------------------------------------------------------------

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device: bool, enable_trace: bool, **kwargs: Any) -> None:
        """Compile the prefill programs on the shapes serving will actually use.

        Deliberately includes a **non-aligned** length: 129 is not a multiple of
        the 32-token page block, of a 32-row tile, or of any power of two, and it
        is the shape the serving path must accept per the goal. Prefill is eager
        here (``enable_trace`` is accepted and ignored -- this port has no
        prefill trace), so the second warmup phase is a no-op.
        """
        if self.already_warmed_up_prefill or enable_trace:
            return
        self.already_warmed_up_prefill = True
        for length in (129, 128):
            self.prefill_forward(
                tokens=torch.zeros((1, length), dtype=torch.int32),
                page_table=self._warmup_page_table(1, length),
                kv_cache=kv_cache,
                enable_trace=False,
                prompt_lens=[length],
                start_pos=[0],
                sampling_params=(self._neutral_sampling_params(1) if can_sample_on_device else None),
            )
        logger.info("Prefill warmup done (lengths 129, 128; 129 is deliberately non-page-aligned)")

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """Compile and then capture the decode traces, at the serving batch.

        Capturing here rather than on the first real token matters twice over:
        it keeps a multi-second trace capture out of the benchmark's
        inter-token latency, and it is the phase the plugin gives us for exactly
        that (phase 1 ``enable_trace=False`` compiles, phase 2 captures).
        """
        if not can_sample_on_device:
            # Host-sampled decode is eager by construction; nothing to capture.
            return
        batch = min(int(max_batch_size), self.max_num_seqs)
        page_table = self._warmup_page_table(batch, self.generator.page_block_size, width=int(num_blocks))
        positions = torch.zeros(batch, dtype=torch.int64)
        self.decode_forward(
            tokens=torch.zeros((batch, 1), dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=positions,
            enable_trace=enable_trace,
            read_from_device=True,
            sampling_params=self._neutral_sampling_params(batch),
            reset_batch=True,
        )
        # The warmup step wrote a token at position 0 of every slot and advanced
        # the device positions; the first real request must not inherit that.
        self._needs_decode_install = True
        self._reset_pending_orders()
        logger.info("Decode warmup done (batch {}, enable_trace={})", batch, enable_trace)

    def _warmup_page_table(self, batch: int, token_count: int, *, width: int | None = None) -> torch.Tensor:
        """A disjoint block assignment for warmup only, at vLLM's table width."""
        width = self.generator.pages_per_user if width is None else int(width)
        blocks = max(1, math.ceil(token_count / self.generator.page_block_size))
        table = torch.zeros((batch, width), dtype=torch.int32)
        for row in range(batch):
            span = min(blocks, width)
            table[row, :span] = torch.arange(row * span, row * span + span, dtype=torch.int32)
        return table

    def _neutral_sampling_params(self, rows: int):
        from vllm_tt_plugin.model_input import TTSamplingParams

        return TTSamplingParams(
            temperature=[0.0] * rows,
            top_k=[1] * rows,
            top_p=[1.0] * rows,
            # The plugin translates its own "no seed" sentinel to ``None`` before
            # the model sees it; the dataclass default of ``0`` is a real seed.
            seed=[None] * rows,
        )

    # -- sampling translation -------------------------------------------------

    def _apply_sampling_params(self, sampling_params, rows: int, *, order=None, graph_rows: int | None = None) -> None:
        """vLLM's per-row sampling request -> the generator's ``(k, p, temp)``.

        Nothing here samples. It only sets the three persistent device parameter
        tensors that ``Qwen3CoderModel.sample_split`` reads, and only when they
        actually changed -- ``set_sampling_params`` no-ops on an identical
        snapshot, so a steady greedy benchmark costs zero host copies per token.
        """
        temps = _as_int_list(getattr(sampling_params, "temperature", None), rows, 0.0)
        top_ks = _as_int_list(getattr(sampling_params, "top_k", None), rows, 1)
        top_ps = _as_int_list(getattr(sampling_params, "top_p", None), rows, 1.0)
        self._audit_unsupported(sampling_params, rows)
        # ``ttnn.sampling``'s per-slot parameters address the *graph*'s rows, so
        # a compacted batch must present them in the same order the rows are in.
        if order is not None:
            picks = [int(v) for v in order.tolist()]
            temps = [temps[i] for i in picks]
            top_ks = [top_ks[i] for i in picks]
            top_ps = [top_ps[i] for i in picks]
        rows = rows if graph_rows is None else int(graph_rows)

        k_out: list[int] = []
        p_out: list[float] = []
        t_out: list[float] = []
        for row in range(rows):
            temperature = float(temps[row])
            top_k = int(top_ks[row])
            top_p = float(top_ps[row])
            if temperature <= 0.0:
                # Greedy. The generator maps temperature 0 to k=1, p=0 itself and
                # then routes to the argmax strategy.
                k_out.append(1)
                p_out.append(0.0)
                t_out.append(0.0)
                continue
            if top_k <= 0 or top_k > SAMPLING_SLOTS:
                # vLLM spells "no top-k" as <=0 and allows any k up to the
                # vocabulary; ``Sampling1DConfig(max_top_k=32)`` is a device
                # limit, so both collapse to the widest supported candidate set.
                if top_k > SAMPLING_SLOTS or top_k <= 0:
                    self._audit["top_k_clamped_requests"] += 1
                    self._warn_once(
                        "top_k",
                        f"top_k={top_k} clamped to {SAMPLING_SLOTS}: the on-device sampler's "
                        "max_top_k is 32 candidates per die-gathered slot.",
                    )
                top_k = SAMPLING_SLOTS
            k_out.append(top_k)
            p_out.append(min(max(top_p, 0.0), 1.0))
            t_out.append(temperature)
        self.generator.set_sampling_params(top_k=k_out, top_p=p_out, temperature=t_out, active_batch=rows)

    def _apply_penalties(
        self, sampling_params, rows: int, prompt_tokens, output_tokens, *, order=None, graph_rows: int | None = None
    ) -> None:
        """vLLM's three penalties -> the generator's staged on-device penalty stage.

        The plugin packs ``presence_penalty`` / ``frequency_penalty`` /
        ``repetition_penalty`` into ``TTSamplingParams`` and sends the token
        history alongside them (``model_runner.py`` populates ``prompt_tokens``
        and ``output_tokens`` "if penalties are needed (decode only)"), because
        ``platform.py`` deliberately does **not** route penalised requests to host
        sampling. This is the model side of that contract; the stage itself is
        ``_WatcherCleanSampling1D._apply_penalties``.

        Neutral on every row is the fast path: ``set_penalty_params`` returns
        False, the ops are not in the captured trace at all, and nothing is
        uploaded.
        """
        presence = _as_int_list(getattr(sampling_params, "presence_penalty", None), rows, 0.0)
        frequency = _as_int_list(getattr(sampling_params, "frequency_penalty", None), rows, 0.0)
        repetition = _as_int_list(getattr(sampling_params, "repetition_penalty", None), rows, 1.0)
        if order is not None:
            picks = [int(v) for v in order.tolist()]
            presence = [presence[i] for i in picks]
            frequency = [frequency[i] for i in picks]
            repetition = [repetition[i] for i in picks]
            # The staged penalty rows are per *graph* row too, and the history
            # they are keyed on has to travel with them.
            prompt_tokens = _reorder_history(prompt_tokens, order, picks)
            output_tokens = _reorder_history(output_tokens, order, picks)
        live, graph_changed = self.generator.set_penalty_params(
            presence=presence,
            frequency=frequency,
            repetition=repetition,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            active_batch=rows if graph_rows is None else int(graph_rows),
        )
        if graph_changed:
            # The mode flip released the decode traces; the next step must
            # reinstall host state rather than replay a freed trace.
            self._needs_decode_install = True
            # Any queued mapping refers to a forward whose output tensor the
            # released trace owned, so those outputs can no longer be read.
            self._reset_pending_orders()
        if live:
            self._audit["penalised_decode_steps"] += 1

    def _audit_unsupported(self, sampling_params, rows: int) -> None:
        """Record -- loudly, once -- the request features this sampler drops."""
        seeds = _as_int_list(getattr(sampling_params, "seed", None), rows, None)
        if any(s is not None for s in seeds):
            self._audit["ignored_seed_requests"] += 1
            self._warn_once(
                "seed",
                "A per-request seed was supplied but this port's sampler draws from its own device "
                "RNG buffer; sampled output is not reproducible from the request seed. See "
                "doc/vllm_integration/README.md, Limitations.",
            )

    # -- prefill --------------------------------------------------------------

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache,
        enable_trace: bool = False,
        prompt_lens=None,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        page_tables_per_layer=None,
        **kwargs: Any,
    ):
        """One serving prefill step, straight into ``generator.prefill_forward``.

        ``tokens`` is ``[num_reqs, max(prompt_lens)]`` with each row's real
        length in ``prompt_lens`` and *garbage past it* -- vLLM slices a shared
        buffer. The generator prefills each row at exactly its own logical
        length (``tokens[user, :prompt_len]``), so a prompt length that is not a
        multiple of the page block, the tile height or any chunk size needs no
        special case: nothing rounds it up on the way in and the selected row is
        ``prompt_len - 1``.
        """
        if page_tables_per_layer is not None:
            raise ValueError("this port has one uniform full-attention KV-cache group; per-layer tables are not used")
        active = int(tokens.shape[0])
        starts = _as_int_list(start_pos, active, 0) if start_pos is not None else [0] * active
        if any(int(p) != 0 for p in starts) and not _PREFIX_CACHING_ENABLED:
            raise ValueError(
                "non-zero prefill start_pos means prefix caching, but it has been "
                "disabled via QWEN3_PREFIX_CACHING=0 while model_capabilities still "
                "advertises supports_prefix_caching=True. Those two must agree: either "
                "unset the kill switch or set supports_prefix_caching=False."
            )
        lengths = [int(n) for n in _as_int_list(prompt_lens, active, int(tokens.shape[1]))]
        device_sampling = sampling_params is not None
        if device_sampling:
            self._apply_sampling_params(sampling_params, active)
            self._audit["device_sampled_prefills"] += 1
        else:
            self._audit["host_sampled_prefills"] += 1

        out = self.generator.prefill_forward(
            tokens.to(torch.int64),
            page_table=self._page_table_for_generator(page_table, active),
            kv_cache=self._require_cache(kv_cache),
            prompt_lens=lengths,
            sampling_mode="device" if device_sampling else "host",
            # A new request is admitted while other slots are mid-decode; see
            # the argument in ``Qwen3CoderGenerator.prefill_forward``, and the
            # measurement behind this default in
            # ``doc/vllm_integration/work_log.md``.
            preserve_decode_traces=PRESERVE_DECODE_TRACES,
            start_pos=starts if _PREFIX_CACHING_ENABLED else None,
        )
        # Whoever held these slots before is gone; the next decode must reinstall
        # host state rather than replay over the device's stale token/position.
        self._needs_decode_install = True
        self._reset_pending_orders()
        if device_sampling:
            return self.generator.read_sampled_tokens(out, active).reshape(active, 1)
        # Host-sampling compatibility mode: vLLM wants ``[B, S, vocab]`` and
        # reads ``[:, -1, :]``. The generator already returns one row per user.
        return out.reshape(active, 1, -1)

    # -- decode ---------------------------------------------------------------

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache,
        start_pos,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        reset_batch: bool | None = None,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        page_tables_per_layer=None,
        **kwargs: Any,
    ):
        """One serving decode step.

        The batch is always the full ``max_num_seqs`` rows -- vLLM pads it so the
        trace shape is constant -- with inactive slots carrying position ``-1``,
        which is the same inactive-row convention the generator's low-level API
        already had.

        Steady state is the whole point: ``tokens``, ``start_pos`` and
        ``page_table`` are all passed as ``None``/unchanged, the two traces
        replay non-blocking, the sampled token is fed back on device and both
        position tensors advance on device. Host work per token is two
        ``ttnn.execute_trace`` calls and one page-table equality check.
        """
        if page_tables_per_layer is not None:
            raise ValueError("this port has one uniform full-attention KV-cache group; per-layer tables are not used")
        caches = self._require_cache(kv_cache)
        rows = int(tokens.shape[0])
        host_tokens = tokens.reshape(-1).to(torch.int64)
        host_positions = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)

        if sampling_params is None:
            # Explicit host-sampling compatibility mode. vLLM routes a request
            # here on its own (logprobs on a 4-die mesh, min_p, bad_words,
            # logit_bias, structured output); it is never the measured path.
            # The eager decode allocates, so ``decode_forward`` releases the
            # captured decode traces -- hence ``_needs_decode_install`` below,
            # which makes the next device-sampled step re-capture through
            # ``_refresh_trace_state`` instead of replaying a released trace.
            self._audit["host_sampled_decode_steps"] += 1
            # Make the demotion loud. vLLM decides this per request and logs
            # nothing, so without this line a served request silently drops from
            # ~49 t/s/u to ~3.6 and the only visible symptom is that the model
            # "got slow" -- which is exactly how this port's batch-scaling defect
            # was first reported. The server-level ``sample_on_device_mode: all``
            # is still correct and still says nothing about it.
            self._warn_once(
                "host_sampled_decode",
                "This request was routed to HOST sampling by vLLM, so decode runs "
                "eager with no captured trace and no width compaction: measured "
                "3.595 t/s/u against 49.345 on the traced path, a ~14x slowdown "
                "for the affected requests. On this 4-die mesh the usual cause is "
                "`logprobs` -- ANY value including 0, because "
                "`model_runner.check_perform_device_sampling` tests "
                "`max_num_logprobs is not None` and then rejects a mesh that is "
                "not 8 or 32 dies. Other triggers are min_p, bad_words, "
                "logit_bias and structured output. Drop the offending parameter "
                "to stay on the traced path; see doc/batch_scaling/README.md, "
                "'logprobs silently cost 14x on this mesh'.",
            )
            self._needs_decode_install = True
            self._reset_pending_orders()
            logits = self.generator.decode_forward(
                host_tokens,
                torch.clamp(host_positions, min=0),
                page_table=self._page_table_for_generator(page_table, rows),
                kv_cache=caches,
                sampling_mode="host",
                enable_trace=False,
                active_batch=rows,
                validate_page_coverage=False,
            )
            return logits.reshape(rows, 1, -1)

        self._audit["device_sampled_decode_steps"] += 1
        install = bool(reset_batch) or self._needs_decode_install

        # The graph width, and with it the graph-row -> vLLM-slot mapping, may
        # only change on an install: that is the one step the plugin guarantees
        # is not steady-decode eligible, so nothing is in flight against the old
        # trace. On every other step the previous mapping still describes the
        # live trace and is reused unchanged.
        previous_order = self._compaction
        if install and len(self._decode_widths) > 1:
            # ``rows`` is the padded decode batch, normally ``max_num_seqs``; a
            # graph can never be wider than the slots the caller actually sent.
            chosen = min(self._choose_width(int((host_positions >= 0).sum())), rows)
            order = self._compaction_order(host_positions, chosen, rows)
            identity = chosen == rows and bool(torch.equal(order, torch.arange(rows)))
            self._compaction = None if identity else order
            if chosen < rows:
                self._audit["narrow_decode_installs"] += 1
            self._audit["decode_graph_width"] = chosen
        width = rows if self._compaction is None else int(self._compaction.numel())
        order = self._compaction
        # With no extra widths configured this stays ``None`` and the generator
        # keeps its own default -- the full configured slot count -- so the
        # shipped path is untouched down to which graph gets captured.
        requested_width = None if self._compaction is None and len(self._decode_widths) == 1 else width

        self._apply_sampling_params(sampling_params, rows, order=order, graph_rows=width)
        # Before the trace is touched: a penalty-mode change releases the decode
        # traces (the ops either are or are not in the captured graph), and the
        # buffers it may allocate cannot be allocated during a capture.
        self._apply_penalties(sampling_params, rows, prompt_tokens, output_tokens, order=order, graph_rows=width)
        # A penalty-mode change releases the traces, so it forces an install even
        # when the scheduler layout did not move. Re-read the flag rather than
        # trusting the value taken before the call; the width decision above is
        # unaffected, because the batch layout is what picks the width and that
        # has not changed.
        install = install or self._needs_decode_install

        if install:
            merged_tokens, merged_positions = self._merge_scheduler_view(
                host_tokens, host_positions, page_table, slot_remap, rows, previous_order
            )
            if order is not None:
                merged_tokens = merged_tokens[order]
                merged_positions = merged_positions[order]
            sampled = self.generator.decode_forward(
                merged_tokens,
                merged_positions,
                page_table=self._compact_page_table(page_table, rows, order),
                kv_cache=caches,
                sampling_mode="device",
                enable_trace=True,
                active_batch=width,
                graph_width=requested_width,
                # Sized once at construction to the served context, so this only
                # asserts the horizon rather than growing anything.
                decode_horizon=self.max_model_len,
                # vLLM's block tables are its own: rows of an unused slot are
                # zero-filled rather than -1, so the standalone disjointness
                # check does not describe them.
                validate_page_coverage=False,
            )
            self._needs_decode_install = False
            self._audit["decode_trace_installs"] += 1
        else:
            sampled = self.generator.decode_forward(
                None,
                None,
                page_table=self._compact_page_table(page_table, rows, order),
                kv_cache=caches,
                sampling_mode="device",
                enable_trace=True,
                active_batch=width,
                graph_width=requested_width,
            )

        # Pair this forward's tokens with the mapping it was issued with, before
        # anything can read them back. ``order`` is ``None`` at full width, which
        # is a meaningful entry: it says "this step needs no un-permutation".
        if self._compaction_enabled:
            self._pending_orders.append((self._next_issue_tag, order))
            self._next_issue_tag += 1
            depth = len(self._pending_orders)
            self._audit["compaction_fifo_max_depth"] = max(self._audit["compaction_fifo_max_depth"], depth)
            if depth > self._pending_orders_cap:
                # Outputs are being issued and never read: the queue is leaking.
                # Fail here rather than let it grow unbounded and mis-pair later.
                raise RuntimeError(
                    f"decode row-mapping queue reached {depth} entries (cap {self._pending_orders_cap}). "
                    "Decode forwards are being issued without their outputs being read, so the "
                    "mapping queue no longer tracks in-flight steps. See "
                    "Qwen3CoderForCausalLM._pending_orders."
                )
        if read_from_device:
            return self.process_decode_output_host(sampled, is_tokens=True)
        return sampled

    def _merge_scheduler_view(
        self,
        host_tokens: torch.Tensor,
        host_positions: torch.Tensor,
        page_table: torch.Tensor,
        slot_remap,
        rows: int,
        previous_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reinstall host state, but keep the device's where the device is right.

        Called only on a layout change, never per token. For a slot that is
        simply continuing, the device already advanced ``current_pos`` past the
        token it just sampled, so ``device_pos`` equals the scheduler's position
        (synchronous scheduling) or is one ahead of it (``--async-scheduling``,
        where vLLM submits step *N+1* before it has applied token *N*). Taking
        the device's pair in both cases is what makes async scheduling safe:
        the host's token would be stale and its position would re-decode a
        position that is already in the cache.

        A slot that changed hands must take the host's pair. Position continuity
        alone cannot tell the two apart -- a recycled slot can coincidentally
        land on a matching position -- so this also requires the slot's page-table
        row to be byte-identical to the one the live trace was captured against.
        A newly admitted request is given fresh physical blocks, so its row moves.
        """
        state = self.generator.decode_device_state()
        if state is None or state["page_table"] is None:
            return host_tokens, host_positions

        # The live trace's rows are *graph* rows. ``previous_order`` says which
        # vLLM slot each one held; scatter them back into slot order before any
        # comparison with the scheduler's view, and leave slots the narrow graph
        # did not cover at the inactive sentinel so they are never "continuing".
        if previous_order is None:
            device_tokens = state["tokens"][:rows].clone()
            device_positions = state["positions"][:rows].clone()
            snapshot = state["page_table"][:rows]
        else:
            covered = previous_order[: state["width"]]
            device_tokens = torch.zeros(rows, dtype=torch.int64)
            device_positions = torch.full((rows,), -1, dtype=torch.int64)
            device_tokens[covered] = state["tokens"][: covered.numel()]
            device_positions[covered] = state["positions"][: covered.numel()]
            snapshot = torch.zeros((rows, state["page_table"].shape[1]), dtype=state["page_table"].dtype)
            snapshot[covered] = state["page_table"][: covered.numel()]
        incoming = torch.as_tensor(page_table).to(torch.int32)[:rows]
        width = min(snapshot.shape[1], incoming.shape[1])

        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1)[:rows].to(torch.int64)
            device_tokens = device_tokens[remap]
            device_positions = device_positions[remap]
            snapshot = snapshot[remap]

        pages_unchanged = torch.all(snapshot[:, :width] == incoming[:, :width], dim=1)
        continuing = (
            ((device_positions == host_positions) | (device_positions == host_positions + 1))
            & (device_positions >= 0)
            & (host_positions >= 0)
            & pages_unchanged
        )
        merged_tokens = torch.where(continuing, device_tokens, host_tokens)
        # ``host_positions`` is the scheduler's view, and the plugin pads rows it
        # is not serving with ``-1`` (``model_runner.py`` pads decode positions
        # with ``-1`` "to indicate no position"). That ``-1`` is exactly the
        # inactive sentinel the traced graph relies on: ``ttnn.plus_one(...,
        # skip_negative_entries=True)`` leaves it alone across replays, and
        # ``_decode_active_mask`` derives the expert-gating mask from
        # ``current_pos >= 0``.
        #
        # Clamping it to 0 here would install an inactive row as position 0, the
        # mask would read it as live, and inactive-row expert gating would
        # silently become a no-op for that slot until the next prefill released
        # the traces. Single-request runs never expose it -- every row is either
        # continuing or genuinely live -- but a server churning 4 of 32 slots
        # would see the gating win appear and disappear with request turnover.
        # So preserve the sentinel and only clamp what is not already a sentinel.
        host_positions_kept = torch.where(host_positions < 0, torch.full_like(host_positions, -1), host_positions)
        merged_positions = torch.where(continuing, device_positions, host_positions_kept)
        return merged_tokens, merged_positions

    # -- async split ----------------------------------------------------------

    def read_decode_output(self, tt_out, async_read: bool = False):
        """Deferred, minimal host read of the sampled-token tensor.

        The payload is one ``[1,1,1,32]`` uint32 tensor -- 128 bytes -- because
        the token was sampled on device. There is no logits readback here and
        there is nothing else to move.
        """
        if isinstance(tt_out, torch.Tensor):
            # Host-sampling compatibility mode already returned host logits.
            return tt_out, []
        if not async_read:
            return tt_out.cpu()
        self._audit["async_decode_reads"] += 1
        self._warn_once(
            "async_split",
            "vLLM took the async decode split: read_decode_output(async_read=True) on a device "
            "handle returned by decode_forward(read_from_device=False). supports_async_decode is "
            "being exercised, not merely declared.",
        )
        host = tt_out.cpu(blocking=False)
        return host, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens: bool = False):
        """Host formatting only; submits no device work.

        Accepts the device handle (the plugin's synchronous path skips
        ``read_decode_output`` entirely), the host ttnn tensor from the async
        path, or an already-host torch tensor.
        """
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        if not is_tokens:
            raise ValueError("host-sampled decode already returns torch logits; nothing to format")
        if ttnn.is_tensor_storage_on_device(tt_out):
            # Only a *device*-resident handle means the plugin skipped
            # ``read_decode_output`` -- its synchronous path -- so the readback
            # happens now, inside ``execute_model``, rather than after the async
            # boundary. Counted so the async/sync split is evidence, not prose.
            #
            # The discriminator has to be device residency, not
            # ``not isinstance(tt_out, torch.Tensor)``: the torch case already
            # returned above, so that test was dead and fired on every step, and
            # the async path's ``read_decode_output`` hands us
            # ``tt_out.cpu(blocking=False)`` -- a ttnn *host* tensor, not a
            # ``torch.Tensor`` -- so async reads were being counted as sync.
            self._audit["sync_decode_reads"] += 1
        tokens = self.generator.read_sampled_tokens(tt_out, self.max_num_seqs)
        # Take the mapping belonging to *this* output, not whatever the adapter
        # currently holds -- see ``_pending_orders``. FIFO is the right pairing
        # because decode forwards are finalized in issue order.
        order = None
        if self._compaction_enabled:
            if not self._pending_orders:
                # There is no safe answer here. Falling back to the adapter's
                # current mapping is precisely the bug the queue exists to
                # prevent, and it would be applied in the one state where the
                # pairing is known to be broken -- every token would go to the
                # wrong request, silently. A crash is strictly better.
                self._audit["compaction_fifo_underflows"] += 1
                raise RuntimeError(
                    "decode output arrived with no queued row mapping. Forwards and host reads are "
                    "no longer paired one-to-one, so the sampled tokens cannot be attributed to "
                    "requests. Refusing to scatter them through a mapping that is not theirs. See "
                    "Qwen3CoderForCausalLM._pending_orders."
                )
            tag, order = self._pending_orders.popleft()
            if tag != self._next_output_tag:
                raise RuntimeError(
                    f"decode row-mapping queue is out of step: popped tag {tag}, expected "
                    f"{self._next_output_tag}. An output has been skipped or read twice, so this "
                    "mapping does not belong to these tokens. See "
                    "Qwen3CoderForCausalLM._pending_orders."
                )
            self._next_output_tag += 1
        if order is not None:
            # Graph row *i* sampled for vLLM slot ``order[i]``. Scatter back;
            # slots the narrow graph did not cover hold no live request and vLLM
            # discards whatever is there.
            restored = torch.zeros(self.max_num_seqs, dtype=tokens.dtype)
            restored[order] = tokens[: order.numel()]
            tokens = restored
        return tokens.reshape(-1, 1)

    # -- helpers --------------------------------------------------------------

    def _require_cache(self, kv_cache):
        """The cache vLLM allocated, and only that one.

        The generator would happily allocate its own on a ``None``; in serving
        that would be a silent second cache that vLLM's block manager knows
        nothing about, so it is an error instead.
        """
        cache = self.kv_cache if kv_cache is None else kv_cache
        if cache is None:
            raise RuntimeError("vLLM has not called allocate_kv_cache; there is no serving cache to use")
        if isinstance(cache, (list, tuple)) and cache and isinstance(cache[0], (list, tuple)):
            raise ValueError("this port is single-submesh; a per-submesh cache list is not expected")
        return cache

    def _compact_page_table(self, page_table, rows: int, order) -> torch.Tensor:
        """vLLM's block table, reordered into graph-row order.

        The page table is the *only* thing that ties a request to its KV pages,
        so permuting its rows is what moves a request between graph rows -- and
        it is why nothing in the cache has to move.
        """
        table = self._page_table_for_generator(page_table, rows)
        return table if order is None else table[order].contiguous()

    def _page_table_for_generator(self, page_table, rows: int) -> torch.Tensor:
        """vLLM's block table at the generator's table width.

        vLLM sizes its table to ``max_num_blocks_per_req``; ``configure_paging``
        already made that the generator's width, so this is normally a no-op.
        Where it is not, pad with **0** rather than the generator's standalone
        ``-1``: the paged decode SDPA kernel rounds its read up to a tile/eight
        page boundary and dereferences every page in the rounded window before
        causal masking, so a tail page must map somewhere valid. vLLM pads its
        own unused entries with 0 for the same reason.
        """
        table = torch.as_tensor(page_table).to(torch.int32)
        if table.ndim != 2:
            raise ValueError(f"page_table must be rank two, got {tuple(table.shape)}")
        target = self.generator.pages_per_user
        if table.shape[1] < target:
            table = torch.nn.functional.pad(table, (0, target - table.shape[1]), value=0)
        elif table.shape[1] > target:
            table = table[:, :target]
        if table.shape[0] < rows:
            table = torch.nn.functional.pad(table, (0, 0, 0, rows - table.shape[0]), value=0)
        return table.contiguous()

    # -- audit ----------------------------------------------------------------

    def serving_audit(self) -> dict:
        """What the serving path actually did, for the stage's fallback audit."""
        audit = dict(self._audit)
        audit["trace_stats"] = dict(self.generator.trace_stats)
        audit["precision_config"] = str(SELECTED_PRECISION_CONFIG)
        audit["max_model_len"] = self.max_model_len
        audit["max_num_seqs"] = self.max_num_seqs
        audit["page_block_size"] = self.generator.page_block_size
        audit["pages_per_user"] = self.generator.pages_per_user
        audit["kv_cache_blocks"] = self.generator.num_blocks
        audit["model_runtime_fallbacks"] = self.model.runtime_fallback_audit(self.max_num_seqs)
        return audit


#: The architecture string in this checkpoint's ``config.json`` is
#: ``Qwen3MoeForCausalLM``; the TT plugin registers models ``TT``-prefixed.
HF_ARCHITECTURE = "Qwen3MoeForCausalLM"
VLLM_ARCHITECTURE = "TT" + HF_ARCHITECTURE

__all__ = ["Qwen3CoderForCausalLM", "HF_ARCHITECTURE", "VLLM_ARCHITECTURE", "HF_MODEL_ID"]
