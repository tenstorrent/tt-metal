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


def _supported_context() -> int:
    try:
        contract = json.loads(CONTEXT_CONTRACT.read_text())
    except (OSError, ValueError):
        return MAX_CONTEXT
    return int(contract.get("current_supported_context") or MAX_CONTEXT)


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
    #: * ``supports_prefix_caching`` -- **False**. Not implemented, not tested;
    #:   the adapter asserts ``start_pos == 0`` on prefill so a future regression
    #:   cannot silently skip cached tokens.
    model_capabilities = {
        "supports_prefix_caching": False,
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

    def _apply_sampling_params(self, sampling_params, rows: int) -> None:
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

    def _apply_penalties(self, sampling_params, rows: int, prompt_tokens, output_tokens) -> None:
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
        live, graph_changed = self.generator.set_penalty_params(
            presence=_as_int_list(getattr(sampling_params, "presence_penalty", None), rows, 0.0),
            frequency=_as_int_list(getattr(sampling_params, "frequency_penalty", None), rows, 0.0),
            repetition=_as_int_list(getattr(sampling_params, "repetition_penalty", None), rows, 1.0),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            active_batch=rows,
        )
        if graph_changed:
            # The mode flip released the decode traces; the next step must
            # reinstall host state rather than replay a freed trace.
            self._needs_decode_install = True
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
        if start_pos is not None and any(int(p) != 0 for p in _as_int_list(start_pos, active, 0)):
            raise ValueError(
                "non-zero prefill start_pos means prefix caching or chunked prefill; "
                "model_capabilities declares neither"
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
        )
        # Whoever held these slots before is gone; the next decode must reinstall
        # host state rather than replay over the device's stale token/position.
        self._needs_decode_install = True
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
            self._needs_decode_install = True
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

        self._apply_sampling_params(sampling_params, rows)
        # Before the trace is touched: a penalty-mode change releases the decode
        # traces (the ops either are or are not in the captured graph), and the
        # buffers it may allocate cannot be allocated during a capture.
        self._apply_penalties(sampling_params, rows, prompt_tokens, output_tokens)
        self._audit["device_sampled_decode_steps"] += 1
        install = bool(reset_batch) or self._needs_decode_install

        if install:
            merged_tokens, merged_positions = self._merge_scheduler_view(
                host_tokens, host_positions, page_table, slot_remap, rows
            )
            sampled = self.generator.decode_forward(
                merged_tokens,
                merged_positions,
                page_table=self._page_table_for_generator(page_table, rows),
                kv_cache=caches,
                sampling_mode="device",
                enable_trace=True,
                active_batch=rows,
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
                page_table=self._page_table_for_generator(page_table, rows),
                kv_cache=caches,
                sampling_mode="device",
                enable_trace=True,
                active_batch=rows,
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

        device_tokens = state["tokens"][:rows].clone()
        device_positions = state["positions"][:rows].clone()
        snapshot = state["page_table"][:rows]
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
