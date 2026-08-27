# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections import defaultdict

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.generator import (
    SDPA_CHUNK_ALIGN,
    ChunkedPrefillPageTableGuardMixin,
    align_num_cached_tokens_to_sdpa,
    max_batched_prefill_users,
    resolve_batched_prefill_chunk_users,
)
from models.demos.gemma4.tt.generator_trace import (
    maybe_disable_pli_prefill_trace,
    patch_gemma4_trace_model_args,
    resolve_gemma4_prefill_chunk_size,
    resolve_gemma4_prefill_trace_enable,
    should_auto_enable_bounded_sliding,
    warmup_gemma4_model_prefill,
)
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES, create_submeshes
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM, allocate_vllm_kv_cache


def _vllm_force_full_isl_single_chunk() -> bool:
    """Opt into full-ISL single-chunk prefill (legacy / debug).

    Default OFF: with tenstorrent/vllm#448 the TT scheduler supports
    token-chunked prefill, so the server mirrors metal's policy chunk path
    (``resolve_gemma4_prefill_chunk_size``, typically 4096 on P150x8 / QB2).

    ``GEMMA4_VLLM_SINGLE_CHUNK`` is the server override and wins when set:
      * ``0``/false → policy multi-chunk (even if ``GEMMA4_DEMO_SINGLE_CHUNK=1``
        leaked into the process env from a prior demo shell)
      * ``1``/true  → force full-ISL single-chunk
    If unset, ``GEMMA4_DEMO_SINGLE_CHUNK=1`` still forces single-chunk for
    demo/server parity experiments.
    """
    vllm_sc = os.environ.get("GEMMA4_VLLM_SINGLE_CHUNK")
    if vllm_sc is not None:
        return vllm_sc.lower() in ("1", "true", "yes")
    return os.environ.get("GEMMA4_DEMO_SINGLE_CHUNK", "0") != "0"


def _full_isl_prefill_chunk_size(max_seq_len: int) -> int:
    """Largest full-ISL single-chunk size that fits ``max_model_len``.

    Prefill pads to the next power-of-2 bucket (``get_padded_prefill_len``).
    For non-pow2 pools (e.g. QB2 31B ``max_context=49152``) the largest valid
    single-chunk length is the floor power-of-2 (32768) — not 49152 (pads to
    65536 > pool) and not ceil-pow2 65536.
    """
    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return 1 << 11
    # Floor power-of-2, minimum 2^11 (smallest prefill bucket used elsewhere).
    floor_pow2 = 1 << max(max_seq_len.bit_length() - 1, 11)
    return min(floor_pow2, max_seq_len)


class _Gemma4VllmOptimizations:
    @staticmethod
    def get_tensor_dtype(decoder_id, tensor, prefetcher=False):
        del decoder_id, tensor, prefetcher
        return ttnn.bfloat16


def _gemma4_prefill_trace_unsafe(model, bounded_sliding_kv_cache) -> bool:
    """True when the hybrid bridge feeds *non-uniform* per-layer page tables
    to the paged ops, so a prefill-trace capture must run *through* the
    per-layer page-table routing rather than the plain ``prefill_forward_text``.

    A direct ``prefill_forward_text`` capture binds the traced paged ops to the
    single full page_table shared by every layer. That only matches runtime
    when every layer truly uses that one table. It diverges — and the captured
    trace then addresses the wrong KV slots, corrupting prefill output —
    whenever:

      * bounded sliding is on and the model has ``sliding_attention`` layers
        (:meth:`_pad_sliding_page_tables_for_bounded` widens only the sliding
        layers, so their table no longer matches the full layers'), or
      * the model kv-shares layers (``kv_shared_layer_map`` re-points a shared
        layer's table at its source's).

    When this returns True, :meth:`warmup_model_prefill` routes the warmup
    capture through :meth:`prefill_forward` (which populates the persistent
    per-layer buffers before capture) — exactly how decode warmup routes
    through ``decode_forward``. Models without sliding layers (or with bounded
    sliding off and no kv-share) can capture directly via
    ``prefill_forward_text``, so the gate is structural and self-scoping
    rather than a hard-coded model list.
    """
    if getattr(model, "kv_shared_layer_map", None):
        return True
    # ``Gemma4Model`` stores the *text* config directly as ``hf_config`` and
    # reads ``self.hf_config.layer_types`` in forward, so look there first;
    # only fall back to a nested ``text_config`` if the top level lacks the
    # field (some unified/multimodal configs nest it).
    hf_config = getattr(model, "hf_config", None)
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        text_config = getattr(hf_config, "text_config", None)
        layer_types = getattr(text_config, "layer_types", None)
    layer_types = list(layer_types or [])
    has_sliding = "sliding_attention" in layer_types
    has_full = "full_attention" in layer_types
    # Mixed sliding + full layers ⇒ vLLM's hybrid kv-cache manager builds
    # multiple kv-cache groups and HMA tensor-sharing packs layers from
    # different groups into one physical KV buffer, indexed by *distinct*
    # per-layer page tables (different block IDs into the shared buffer). The
    # prefill-trace warmup captures a single broadcast table for every layer,
    # so shared layers collide on the same slots and corrupt the KV cache on
    # replay — independent of bounded sliding. Bounded sliding adds further
    # per-layer width divergence (sliding tables padded to the window) on top.
    if has_sliding and has_full:
        return True
    if bounded_sliding_kv_cache and has_sliding:
        return True
    return False


def _resolve_vllm_bounded_sliding(max_seq_len, mesh_device, model_path, *, hybrid_groups_enabled: bool) -> bool:
    """Mirror demo: auto policy + ``GEMMA4_BOUNDED_SLIDING_KV_CACHE`` / legacy env."""
    # Hybrid-groups mode historically defaulted bounded ON; keep that unless env overrides.
    _bounded_default = "1" if hybrid_groups_enabled else None
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE")
    if _bs_env is None and _bounded_default is not None:
        _bs_env = _bounded_default
    if _bs_env is None:
        # Also accept GEMMA4_BOUNDED_SLIDING (demo alias) when unset.
        _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    if _bs_env is None:
        return should_auto_enable_bounded_sliding(max_seq_len, mesh_device, model_path)
    return _bs_env.lower() in ("1", "true", "yes")


def _patch_model_args(
    model_args,
    mesh_device,
    max_batch_size,
    max_seq_len,
    model_path,
    prefill_trace_enabled=True,
    *,
    bounded_sliding=False,
):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    # Prefill chunking (two cooperating layers after tenstorrent/vllm#448):
    #   1) vLLM scheduler token-chunked prefill (enable_chunked_prefill +
    #      max_num_batched_tokens / long_prefill_token_threshold)
    #   2) metal generator max_prefill_chunk_size via policy
    #      (resolve_gemma4_prefill_chunk_size: default prefill_chunk, then
    #      optional prefill_chunk_by_isl for high-ISL tiers)
    #
    # Keep llm.yaml scheduler tokens aligned with the resolved metal chunk for
    # the configured max_context. Override with GEMMA4_GEN_PREFILL_CHUNK, or
    # force full-ISL single-chunk via GEMMA4_VLLM_SINGLE_CHUNK=1.
    chunk_override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
    if chunk_override > 0:
        model_args.max_prefill_chunk_size = chunk_override
        logger.info(
            "Gemma4 vLLM: GEMMA4_GEN_PREFILL_CHUNK={} → max_prefill_chunk_size={}",
            chunk_override,
            model_args.max_prefill_chunk_size,
        )
    elif _vllm_force_full_isl_single_chunk():
        model_args.max_prefill_chunk_size = _full_isl_prefill_chunk_size(max_seq_len)
        logger.info(
            "Gemma4 vLLM: full-ISL single-chunk prefill "
            "(max_prefill_chunk_size={}, max_seq_len={}). "
            "Unset GEMMA4_VLLM_SINGLE_CHUNK to use policy multi-chunk.",
            model_args.max_prefill_chunk_size,
            max_seq_len,
        )
    else:
        model_args.max_prefill_chunk_size = resolve_gemma4_prefill_chunk_size(
            max_seq_len,
            mesh_device=mesh_device,
            # Measured P150x8 / QB2 policies supply default 4096 (+ ISL tiers);
            # other boards keep full-ISL until validated (same as demo).
            non_qb2_default=max_seq_len,
            model_name_or_path=model_path,
            bounded_sliding=bounded_sliding,
        )
        logger.info(
            "Gemma4 vLLM: policy prefill chunk={} "
            "(max_seq_len={}, bounded_sliding={}) — align vLLM "
            "max_num_batched_tokens / long_prefill_token_threshold to this value",
            model_args.max_prefill_chunk_size,
            max_seq_len,
            bounded_sliding,
        )
    patch_gemma4_trace_model_args(model_args, prefill_trace_enabled=prefill_trace_enabled)
    model_args.optimizations = _Gemma4VllmOptimizations()
    model_args.mesh_device = mesh_device
    model_args._gemma4_model_path = model_path
    model_args.is_llama_vision = lambda: False


class Gemma4ForCausalLM(ChunkedPrefillPageTableGuardMixin, HybridAttentionForCausalLM):
    """Gemma4 — hybrid attention (sliding-window + full).

    Gemma4's decoder alternates ``sliding_attention`` and ``full_attention``
    layers per ``hf_config.layer_types``, so the bridge inherits from
    :class:`HybridAttentionForCausalLM` to opt into vLLM's hybrid kv cache
    manager. ``get_kv_cache_spec`` is inherited; layer-routed page tables
    flow through the model's ``_active_page_tables_per_layer`` stash and
    are picked up inside ``Gemma4Model.{ttnn_prefill_forward,
    ttnn_decode_forward}`` (mirrors the gpt-oss bridge).
    """

    # Async decode closes the ~15–20% metal↔server B=1 gap (#51186): with
    # ``async_scheduling`` the plugin overlaps CPU scheduling with the previous
    # device step via ``decode_forward(read_from_device=False)`` +
    # ``read_decode_output(async_read=True)`` (inherited from ``Generator``).
    # Requires on-device token feedback + position plus_one (non-PLI only;
    # see ``Gemma4Model._tt_vllm_always_refresh_decode_trace_inputs``).
    #
    # Default ON for non-PLI. Token-doubling under async is mitigated by
    # ``merge_async_ahead_decode_tokens`` + vLLM preempt bookkeeping. Kill-switch:
    # ``GEMMA4_SUPPORTS_ASYNC_DECODE=0``. PLI models force False in ``__init__``.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": os.environ.get("GEMMA4_SUPPORTS_ASYNC_DECODE", "1").lower() in ("1", "true", "yes"),
        # Gemma4ModelArgs exposes no get_attn_sdpa_program_config, so Generator
        # cannot derive the resume offset alignment and must be told it. Same pin
        # align_num_cached_tokens_to_sdpa applies locally.
        "resumed_prefill_token_alignment": SDPA_CHUNK_ALIGN,
        "supports_sample_on_device": True,
        # prefill_forward_text routes a nonzero start_pos to the chunked SDPA and
        # floors the offset to resumed_prefill_token_alignment, so a prompt split
        # across engine steps needs no new prefill code.
        "supports_chunked_prefill": True,
    }

    # vLLM pads decode to the nearest of these (not always max_num_seqs) so B=1
    # recovers the metal demo SDPA/matmul path (~27 tok/s/user vs ~20 at B=32).
    #
    # This is the *candidate* set. ``warmup_model_decode`` narrows it in place to
    # the buckets it actually captured a decode trace for, because the plugin
    # treats this attribute as the whole contract: padding to a bucket with no
    # captured trace leaves the device in an undefined state (on Wormhole a
    # fast-dispatch hang). Do not publish a wider list than what is warmed.
    tt_supported_decode_batch_sizes = SUPPORTED_PREFILL_BATCH_SIZES

    # Set True once ``warmup_model_decode`` has captured its traces. Previously
    # the presence of ``tt_warmed_decode_batch_sizes`` doubled as this sentinel;
    # that attribute is gone, so the signal is now explicit.
    _decode_warmup_complete = False

    # Hybrid vLLM kv-cache groups: env-gated via ``GEMMA4_HYBRID_KV_CACHE_GROUPS``
    # (default OFF). Toggle from the tt-inference-server model-spec env so the KV
    # mode is config-driven and reversible without a code change.
    #
    # OFF (default): ``get_kv_cache_spec`` emits ``FullAttentionSpec`` for *every*
    # layer, which vLLM merges into a single ``UniformTypeKVCacheSpecs`` group, so
    # the whole block pool backs each request and the full ``max_model_len`` is
    # admissible (verified ~100K ISL). Sliding layers allocate full-length KV
    # unless ``bounded_sliding`` is on — then :meth:`allocate_kv_cache_per_layer`
    # shrinks them to ``sliding_window/block_size * max_batch`` (demo parity) and
    # :meth:`_pad_sliding_page_tables_for_bounded` remaps page tables to dense
    # local block IDs. Without that shrink, auto-bounded 256k OOMs on 31B.
    #
    # ON (``GEMMA4_HYBRID_KV_CACHE_GROUPS=1``): sliding layers emit
    # ``SlidingWindowSpec`` and form their own kv_cache_groups, so the 40 sliding
    # layers only allocate the 1024-token window (``cache_position_modulo`` bounded
    # ring on device) — far less KV DRAM, higher concurrency/throughput. Tradeoffs:
    # vLLM splits the block pool across groups, so a single request is capped at
    # ~``num_blocks // num_groups`` tokens (long-context admission regresses), and
    # bounded sliding's known >~34k degradation applies. Bounded sliding is tied to
    # this flag (below). This is the pre-#48283 path, restored behind the env gate.
    #
    # KNOWN BLOCKER (why ON is not the default yet): the hybrid path serves
    # correctly up to ISL 4096 — including the single-user 2048 prefill that used
    # to hang (#49083) — but crashes at ISL >= 8192. The full-attention layers'
    # long-context chunked-prefill SDPA
    # (``ttnn.transformer.chunked_scaled_dot_product_attention``) TT_FATALs on
    # ``k_shape[3] == DH``: under the shared kv-cache group the full-attn K/V is
    # stored at the sliding head_dim (256) while full attention needs DH=512. The
    # non-chunked paged ops reconcile this via the ``effective_block_size`` override
    # (see attention/operations.py), but the chunked SDPA op takes no such block/
    # head_dim knob — fixing it (an op/kernel change, or allocating full-attn its
    # own head_dim buffer) is the remaining work to make ON viable end-to-end.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = os.environ.get("GEMMA4_HYBRID_KV_CACHE_GROUPS", "0") != "0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prefer the flag baked into the TT model at create time (resolve + env);
        # fall back to hybrid-default env when the model was built elsewhere.
        model0 = self.model[0] if getattr(self, "model", None) else None
        if model0 is not None and hasattr(model0, "bounded_sliding_kv_cache"):
            self._bounded_sliding_kv_cache = bool(model0.bounded_sliding_kv_cache)
        else:
            _bounded_default = "1" if self._HYBRID_KV_CACHE_GROUPS_ENABLED else "0"
            self._bounded_sliding_kv_cache = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE", _bounded_default) != "0"
        # PLI models must restage decode inputs from host every step; async lag
        # would restage a stale token. Force capability off even if env default
        # is on (platform then disables async_scheduling).
        if model0 is not None and (
            bool(getattr(model0, "hidden_size_per_layer_input", 0))
            or bool(getattr(model0, "_tt_vllm_always_refresh_decode_trace_inputs", False))
        ):
            self.model_capabilities = {
                **self.model_capabilities,
                "supports_async_decode": False,
            }
        # On-device decode sampling only exists when Gemma4Model actually built a
        # SamplingGenerator. That needs the per-device vocab shard to fit ttnn's
        # 64k topk width: Gemma4's vocab is 262144, so TP>=4 shards to <=65536 and
        # works, but TP=2 (WH N300 1x2) shards to 131072 and Gemma4Model leaves
        # ``self.sampling = None``. Advertising the capability anyway made vLLM
        # request on-device decode logits and the engine died on the
        # ``self.sampling is not None`` assert in ``ttnn_decode_forward``. Report
        # what the model can really do so the platform falls back to host
        # sampling (which all-gathers the full vocab and is correct, just slower).
        if model0 is not None and not bool(getattr(model0, "_supports_on_device_sampling", True)):
            self.model_capabilities = {
                **self.model_capabilities,
                "supports_sample_on_device": False,
            }
            _tp = getattr(getattr(model0, "mesh_config", None), "tp", "?")
            logger.info(
                f"Gemma4: on-device sampling unavailable on this mesh (tp={_tp}, "
                f"vocab={getattr(model0, 'vocab_size', '?')} shards wider than ttnn topk's 64k); "
                "advertising supports_sample_on_device=False so decode samples on host."
            )
        # Host-side TTFT / decode tok/s for metal↔server parity checks.
        # Compare these to demo ``inference_prefill`` / decode tok/s/user logs.
        self._perf_decode_tokens = 0
        self._perf_decode_s = 0.0
        self._perf_log_every = max(1, int(os.environ.get("GEMMA4_VLLM_PERF_LOG_EVERY", "32")))
        # Batch-keyed decode traces (mixin); must exist before first sample.
        self._prev_decode_batch = None
        # Extra synchronize_device stalls async overlap (#51186). Default off;
        # set GEMMA4_VLLM_DECODE_SYNC_EVERY=log/1 for sync wall-clock parity.
        _sync_default = "0" if self.model_capabilities.get("supports_async_decode") else "log"
        self._perf_decode_sync_every = os.environ.get("GEMMA4_VLLM_DECODE_SYNC_EVERY", _sync_default)

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        """Warm decode traces at B=1 and B=max (trace-region friendly).

        Full power-of-two warmup (1..32) would multiply resident decode-trace
        DRAM (~360MB at B=32 alone). B=1 recovers metal short-ISL tok/s; B=max
        covers concurrency. Intermediate active counts pad up to B=max.
        Override with ``GEMMA4_DECODE_WARMUP_BATCHES=1,8,32``.
        """
        max_b = int(max_batch_size)
        override = os.environ.get("GEMMA4_DECODE_WARMUP_BATCHES")
        if override:
            sizes = sorted({int(x) for x in override.split(",") if x.strip() and int(x) <= max_b})
        else:
            sizes = sorted({1, max_b} if max_b > 1 else {1})
        # Restrict to declared supported buckets.
        supported = set(self.tt_supported_decode_batch_sizes)
        sizes = [b for b in sizes if b in supported or b == max_b]
        if not sizes:
            sizes = [max_b]
        # Narrow the declared buckets to exactly what we warm below, so the
        # plugin never pads to a bucket without a captured decode trace. This
        # replaces the previous separate ``tt_warmed_decode_batch_sizes``
        # attribute — an unwarmed bucket is simply not supported (review on
        # tenstorrent/vllm#455).
        self.tt_supported_decode_batch_sizes = tuple(sizes)
        self._decode_warmup_complete = True
        # Smallest → largest so the final batch leaves sampling traces bound to
        # max_batch logits (sampling capture is skipped for smaller buckets).
        for batch in sorted(sizes):
            # Drop sampling traces from the previous batch — they bind logits
            # tensor identity/batch and will ValueError on the next size.
            for m in self.model:
                sampling = getattr(m, "sampling", None)
                if sampling is not None and hasattr(sampling, "reset_trace"):
                    sampling.reset_trace()
            logger.info("Gemma4 vLLM: decode warmup batch_size={}", batch)
            super().warmup_model_decode(
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                max_batch_size=batch,
                num_blocks=num_blocks,
                can_sample_on_device=can_sample_on_device,
                **kwargs,
            )

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", **kwargs) -> int:
        # The all-user KV-cache pool size is a per-device / per-model tuning knob,
        # not a model constant: with hybrid KV groups disabled every layer
        # allocates a full-length KV buffer, so the pool that fits in DRAM is
        # hardware-specific (e.g. ~49K on QB2/P300x2 for 31B, ~131K for 12B).
        # Keep that value OUT of the model code — set ``GEMMA4_MAX_TOKENS_ALL_USERS``
        # from the tt-inference-server model spec's per-device ``env_vars`` block
        # (gated there by device + model). This generic, value-free hook just
        # honors that override and otherwise defers to the default.
        override = os.environ.get("GEMMA4_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        return super().get_max_tokens_all_users(model_name=model_name, **kwargs)

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        return maybe_disable_pli_prefill_trace(enable_trace, self.model[0], batch_size=batch_size)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
    ):
        # #49083 fix: pre-capture the prefill-bucket traces here, at warmup,
        # rather than lazily on the first runtime prefill. A cold *eager*
        # prefill dispatched after a shared-Generator traced-decode session
        # (the release workflow's evals phase) wedges the fetch queue
        # (nlp_concat_heads) at ISL=2048 — capturing every bucket up front so
        # runtime only *replays* removes that trace->eager transition.
        #
        # The hybrid per-layer page tables diverge from the single broadcast
        # table a direct ``prefill_forward_text`` capture would bind, so route
        # the capture through ``prefill_forward`` (``prefill_forward_fn`` below).
        # That sets up per-layer routing and populates the persistent per-layer
        # buffers *before* the traced forward, so the captured paged ops bind
        # those buffers — identical to how decode warmup binds via
        # ``decode_forward``. Runtime ``prefill_forward`` then just refreshes the
        # same buffers' block IDs out-of-trace and replays. ``_mock_tokens``
        # sizes the warmup page table to the runtime width, so the persistent
        # buffers match runtime (and decode-warmup) shapes.
        #
        # GEMMA4_DISABLE_PREFILL_TRACE=1 keeps prefill fully eager (no capture).
        # Bounded sliding: never capture prefill TRACE — mid-forward paged_fill
        # corrupts token-0 on TP (see resolve_gemma4_prefill_trace_enable).
        prefill_forward_fn = None
        if self._bounded_sliding_kv_cache:
            enable_trace = False
        elif enable_trace and _gemma4_prefill_trace_unsafe(self.model[0], self._bounded_sliding_kv_cache):
            prefill_forward_fn = self.prefill_forward
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
            prefill_forward_fn=prefill_forward_fn,
        )

    def prefill_forward_text(self, *args, enable_trace=True, **kwargs):
        tokens = args[0] if args else kwargs.get("tokens")
        batch_size = tokens.shape[0] if tokens is not None else 1
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)
        if tokens is not None:
            batch_seq_len = tokens.shape[1]
            prompt_lens = kwargs.get("prompt_lens")
            start_pos = kwargs.get("start_pos")
            prompt_lens_list = prompt_lens if prompt_lens is not None else [batch_seq_len] * batch_size
            if not isinstance(prompt_lens_list, list):
                prompt_lens_list = prompt_lens_list.tolist()
            num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
            if start_pos is not None:
                num_cached_per_user = align_num_cached_tokens_to_sdpa(num_cached_per_user)
                kwargs["start_pos"] = num_cached_per_user
                start_pos = num_cached_per_user
            prefill_seq_lens = [
                get_padded_prefill_len(seq_len - num_cached)
                for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
            ]
            page_table = kwargs.get("page_table")
            # Hetero *actual* lens OK: per-slot valid_seq_lens caps KV fill.
            can_batch_prefill = (
                page_table is not None
                and batch_size > 1
                and len(set(prefill_seq_lens)) == 1
                and self.data_parallel == 1
                and not getattr(self.model_args[0], "disable_batched_prefill", False)
                and all(n == 0 for n in num_cached_per_user)
            )
            enable_trace = resolve_gemma4_prefill_trace_enable(
                enable_trace,
                self.model[0],
                self.model_args[0],
                batch_size=batch_size,
                prefill_seq_lens=prefill_seq_lens,
                can_batch_prefill=can_batch_prefill,
                empty_slots=kwargs.get("empty_slots"),
            )
        return super().prefill_forward_text(*args, enable_trace=enable_trace, **kwargs)

    def _bounded_sliding_min_page_table_cols(self, kv_cache) -> int | None:
        """Min page-table columns so ``cache_position_modulo`` fits the kernel check.

        ``paged_fill_cache`` requires ``modulo <= effective_block_size * cols``.
        For sliding layers the kernel's block_size is the cache's declared
        ``shape[2]`` (typically 64) — not the HMA-scaled effective size used
        for full-attn views. Floor at ``cdiv(sliding_window, block_size)``.
        """
        if not self._bounded_sliding_kv_cache or kv_cache is None:
            return None
        sliding_window = getattr(self._text_config(), "sliding_window", None)
        if sliding_window is None:
            return None
        try:
            block_size = int(kv_cache[0][0].shape[2])
        except (TypeError, IndexError, AttributeError):
            return None
        if block_size <= 0:
            return None
        from models.tt_transformers.tt.common import num_blocks_in_seq

        return num_blocks_in_seq(int(sliding_window), block_size)

    def _get_prefill_user_page_table(
        self,
        page_table,
        kv_cache,
        prefill_len,
        trace_enabled=False,
        prefill_seq_len=None,
        use_batched_prefill=False,
        user_id=None,
        padded_batch_size=None,
        use_full_prompt_len=False,
    ):
        """Override the shared Generator helper to size/slice the
        per-user page table to the *smallest* effective block_size in
        the model, not the cache's declared block_size.

        Background: ``Generator._get_prefill_user_page_table`` slices
        the page_table to ``cdiv(prefill_seq_len, get_block_size(kv_cache))``
        columns, where ``get_block_size`` reads ``kv_cache[0][0].shape[2]``
        — the declared block_size of layer 0's K cache. Under vLLM's
        hybrid kv-cache-groups manager that's the buffer's *allocation*
        block_size, which for Gemma4-E2B is sliding's 128 (sliding
        layers come first in the layer order and their spec wins the
        shared buffer's shape). But full-attention layers operate
        through a view with effective block_size=64 (see
        ``attention/{prefill,decode}.py``), and ``paged_fill_cache``
        validates ``input_seq_len <= max_num_blocks_per_seq *
        effective_block_size``. With the legacy slice the full layer
        sees too few blocks and the validator fires.

        Use the smallest effective block_size across all attention
        layers — same invariant the warmup ``_mock_tokens`` override
        uses — so the slice covers every layer's needs.

        Bounded sliding: never slice below ``sliding_window/block_size``
        columns (batched or not). Batched warmup at seq=128 otherwise
        yields 2 columns and TT_FATALs ``cache_position_modulo`` (1024).
        """
        import torch

        from models.tt_transformers.tt.common import num_blocks_in_seq

        min_bounded_cols = self._bounded_sliding_min_page_table_cols(kv_cache)

        # vLLM token-chunked prefill / APC: ``prefill_len`` is the full prompt
        # while ``prefill_seq_len`` is only the padded *current* chunk. Truncating
        # to the chunk width leaves continuation ``chunk_page_table`` slices as
        # all -1 → ``paged_fill_cache`` skips writing tokens past the first
        # scheduler chunk (LB 12B ~9k coherence cliff, #51186). Keep the full
        # mapping whenever the prompt is longer than this chunk's pad.
        # ``prefill_len`` is a list under batched prefill — only compare scalars.
        if (
            not use_full_prompt_len
            and prefill_seq_len is not None
            and prefill_len is not None
            and not isinstance(prefill_len, (list, tuple))
            and int(prefill_len) > int(prefill_seq_len)
        ):
            use_full_prompt_len = True

        if use_batched_prefill:
            from models.tt_transformers.tt.common import get_block_size

            block_size = get_block_size(kv_cache)
            batch_dim = padded_batch_size if padded_batch_size is not None else self.model_args[0].max_batch_size
            # Batched path always sizes to the padded chunk grid (slot layout).
            num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            if min_bounded_cols is not None:
                num_blocks = max(num_blocks, min_bounded_cols)
            if page_table.shape[1] < num_blocks:
                pad = torch.zeros(
                    (page_table.shape[0], num_blocks - page_table.shape[1]),
                    dtype=torch.int32,
                )
                page_table = torch.cat([page_table, pad], dim=1)
            page_table = page_table[:, :num_blocks]
            if trace_enabled and page_table.shape[1] < num_blocks:
                padding = torch.zeros(page_table.shape[0], num_blocks - page_table.shape[1], dtype=torch.int32)
                page_table = torch.cat([page_table, padding], dim=1)
            padded_page_table = torch.zeros(batch_dim, page_table.shape[1], dtype=torch.int32)
            assert user_id is not None
            for i, user in enumerate(user_id):
                padded_page_table[user, :] = page_table[i, :]
            return padded_page_table

        # Per-user (non-batched) path: replicate the base behavior but
        # with effective block_size instead of ``get_block_size``.
        cache = kv_cache[0][0]  # layer 0, K (HMA-shared across specs)
        cache_block_size = cache.shape[2]
        cache_head_dim = cache.shape[-1]
        head_dims = {layer.self_attn.config.head_dim for layer in self.model[0].layers}
        max_head_dim = max(head_dims)
        effective_block_size = cache_block_size * cache_head_dim // max_head_dim

        if use_full_prompt_len:
            target_prefill_len = prefill_len
        else:
            target_prefill_len = prefill_seq_len if prefill_seq_len is not None else prefill_len
        num_blocks = num_blocks_in_seq(target_prefill_len, effective_block_size)
        if min_bounded_cols is not None:
            num_blocks = max(num_blocks, min_bounded_cols)
        if page_table.shape[1] < num_blocks:
            padding = torch.zeros(1, num_blocks - page_table.shape[1], dtype=torch.int32)
            page_table = torch.cat([page_table, padding], dim=1)
        return page_table[:, :num_blocks]

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        """Override warmup page_table sizing for the hybrid-kv-cache-groups
        path.

        Warmup must produce a page_table whose shape matches the *runtime*
        legacy page_table shape (``model_input.block_tables`` =
        ``block_tables_per_group[0]`` in the plugin), because the decode
        trace captures device tensors at warmup shapes and ``copy_host_to_device``
        asserts shape-equality on every replay. The runtime per-group
        block_table for layer 0's group has width
        ``cdiv(max_model_len, group_block_size_after_unification)``; for
        Gemma4-E2B layer 0 is sliding and the unifier doubled sliding's
        block_size from ``cache_config.block_size`` to match the larger
        full-attn page size (sliding head_dim=256 → 128 block_size;
        full head_dim=512 → 64 block_size). The cache tensor's declared
        ``shape[2]`` is that post-unification block_size, so reading
        directly from layer 0's K-cache shape gives the right value.

        The full-attention layers operate through a view with the smaller
        effective block_size (= ``cache.shape[2] * cache.shape[-1] // full_head_dim``
        = 64), but their per-layer block_table is *padded* to the same width
        by the plugin's ``_block_tables_per_layer`` — so a single warmup
        width still aligns every layer's persistent buffer. The smaller
        effective block_size narrows the kernel's ``input_seq_len <=
        max_num_blocks_per_seq * block_size`` validation budget; warmup
        chunks stay well under that limit, and full-coverage of
        ``max_model_len`` for full-attn would require sizing the per-layer
        table separately (separate work).
        """
        import torch

        from models.tt_transformers.tt.common import num_blocks_in_seq

        ret = {
            "tokens": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "prompt_lens": torch.tensor([seq_len] * batch_size, dtype=torch.long),
            "empty_slots": list(range(batch_size)),
        }

        page_table_warmup = None
        if kv_cache is not None and kv_cache[model_id] is not None:
            cache = kv_cache[model_id][0][0]  # layer 0, K
            cache_block_size = cache.shape[2]
            # Match the plugin's runtime page_table width for layer 0's
            # group: ``cdiv(max_seq_len, declared_block_size)``.
            max_seq_len = self.model_args[model_id].max_seq_len
            num_blocks = num_blocks_in_seq(max_seq_len, cache_block_size)
            page_table_warmup = torch.zeros(batch_size, num_blocks, dtype=torch.int32)

        ret["page_table"] = page_table_warmup
        return ret

    # ── vLLM ``VllmModelForTextGeneration`` protocol shim ────────────────
    #
    # vLLM's ``is_text_generation_model`` predicate checks for
    # ``embed_input_ids``, ``forward(input_ids, positions)``, and
    # ``compute_logits`` on the resolved model class — that's how upstream
    # ``runner_type=="generate"`` validates a model is generative. Other TT
    # models (Gemma3, GptOss, etc.) get away without these because vLLM
    # finds an upstream torch implementation in its registry first and uses
    # *that* class for inspection, while the plugin's ``TT``-prefix logic
    # routes execution to the TT class. Gemma4 has no upstream vLLM impl,
    # so the inspection has to land on this class.
    #
    # Actual execution on the TT path goes through ``prefill_forward`` /
    # ``decode_forward`` (called by the TT runner via the
    # ``HybridAttentionForCausalLM`` overrides above), so these stubs are
    # never invoked. They exist purely to satisfy the protocol check.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; embeddings happen on TT via "
            "prefill_forward / decode_forward, not through this method."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; the TT runner invokes "
            "prefill_forward / decode_forward, not forward()."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; logits are produced on TT "
            "and surfaced through prefill_forward / decode_forward."
        )

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Build per-layer KVCacheSpec, honoring Gemma4's per-layer-type
        differences in ``head_dim`` and ``num_kv_heads``.

        The base ``HybridAttentionForCausalLM.get_kv_cache_spec`` assumes
        all layers share one ``head_size`` / ``num_kv_heads`` (only the
        sliding-vs-full *spec class* changes). That's true for Gemma3 but
        not Gemma4: sliding layers use ``head_dim`` (256 on E2B/E4B),
        full layers use ``global_head_dim`` (512). Sliding and full also
        each have their own ``num_key_value_heads`` (with the full count
        falling back to the sliding count when ``num_global_key_value_heads``
        is unset). Emitting one uniform spec made the K tensor produced
        by full-attention layers mismatch the cache shape and trip
        ``Last dim of input tensor must match last dim of cache tensor``
        in ``paged_update_cache``.

        vLLM's hybrid kv cache manager handles the resulting
        non-uniform-shape grouping fine (sliding layers form one group,
        full layers another), so the only thing that needs to differ
        between groups is the spec — block_size stays uniform.
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(
                f"{cls.__name__}.get_kv_cache_spec requires "
                "hf_config.text_config.layer_types (one of 'full_attention' / "
                "'sliding_attention' per layer); none found on this model"
            )

        sliding_kv_heads = text_config.num_key_value_heads
        sliding_head_dim = text_config.head_dim
        sliding_window = getattr(text_config, "sliding_window", None)
        full_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or sliding_kv_heads
        full_head_dim = getattr(text_config, "global_head_dim", None) or sliding_head_dim

        tp = parallel_config.tensor_parallel_size
        sliding_kv_heads_per_dev = sliding_kv_heads // tp
        full_kv_heads_per_dev = full_kv_heads // tp

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size

        spec_per_layer = {}
        for i, lt in enumerate(layer_types):
            name = f"model.layers.{i}.self_attn"
            if lt == "sliding_attention":
                if sliding_window is None:
                    raise ValueError(
                        f"layer_types[{i}] is 'sliding_attention' but "
                        f"hf_config.sliding_window is None on {cls.__name__}"
                    )
                if cls._HYBRID_KV_CACHE_GROUPS_ENABLED:
                    # Hybrid ON: windowed ``SlidingWindowSpec`` so sliding layers
                    # form their own kv_cache_group(s) and only allocate the
                    # bounded window on device (memory-efficient; see the class
                    # docstring for the single-request ISL-cap tradeoff).
                    spec_per_layer[name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=sliding_kv_heads_per_dev,
                        head_size=sliding_head_dim,
                        dtype=dtype,
                        sliding_window=sliding_window,
                    )
                else:
                    # Hybrid OFF: ``FullAttentionSpec`` for sliding layers too,
                    # keeping their own (sliding) num_kv_heads/head_size. vLLM
                    # merges into one ``UniformTypeKVCacheSpecs`` group so the
                    # full block pool backs every request. Specs still declare
                    # full ``num_blocks``; when bounded_sliding is on,
                    # allocate_kv_cache_per_layer shrinks the physical sliding
                    # buffers and remaps page tables (demo parity).
                    spec_per_layer[name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=sliding_kv_heads_per_dev,
                        head_size=sliding_head_dim,
                        dtype=dtype,
                    )
            elif lt == "full_attention":
                spec_per_layer[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=full_kv_heads_per_dev,
                    head_size=full_head_dim,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    f"Unsupported layer_type {lt!r} at layer {i} on "
                    f"{cls.__name__}; expected 'full_attention' or "
                    "'sliding_attention'"
                )
        return spec_per_layer

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance", "accuracy"):
            raise ValueError("Gemma4 TT optimization profiles: None|performance|accuracy " f"(got {optimizations!r})")
        # ``accuracy`` is accepted for API parity with tt_transformers but must
        # NOT enable linear HiFi4/fp32 — that caused unicode garbage on LB 12B.
        # Production GeLU is always Accurate (see ``compute_config.gelu_variant``).
        if optimizations == "accuracy":
            logger.info(
                "Gemma4 optimizations=accuracy: keeping production defaults "
                "(GeLU=Accurate; no linear HiFi4/fp32 — those regress decode)"
            )

        model_path = hf_config._name_or_path
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        # Bounded sliding: mirror demo (auto policy + env). Hybrid-groups mode
        # still defaults ON when env unset — see ``_resolve_vllm_bounded_sliding``.
        bounded_sliding_kv_cache = _resolve_vllm_bounded_sliding(
            max_seq_len,
            mesh_device,
            model_path,
            hybrid_groups_enabled=cls._HYBRID_KV_CACHE_GROUPS_ENABLED,
        )

        model_args = []
        model = []
        state_dict = None
        for submesh in submesh_devices:
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                dtype=ttnn.bfloat16,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                paged_attention_config=None,
                create_kv_cache=False,
                model_path=model_path,
                bounded_sliding_kv_cache=bounded_sliding_kv_cache,
            )
            prefill_trace_unsafe = _gemma4_prefill_trace_unsafe(model_i, bounded_sliding_kv_cache)
            # GH #49083 fix: pre-capture the prefill device traces at *warmup*
            # (see ``warmup_model_prefill``), routed through ``prefill_forward``
            # for the hybrid per-layer case. A cold *eager* prefill dispatched
            # after a shared-Generator traced-decode session (the release
            # workflow's evals phase) wedges the fetch queue (nlp_concat_heads)
            # at ISL=2048; capturing every bucket up front so runtime only
            # *replays* removes that trace->eager transition. Capturing at warmup
            # (before any traced decode) is what makes it safe — a lazy
            # first-runtime capture still hits the wedge.
            # GEMMA4_DISABLE_PREFILL_TRACE=1 restores the fully-eager prefill.
            prefill_trace_enabled = os.environ.get("GEMMA4_DISABLE_PREFILL_TRACE", "0") != "1"
            if prefill_trace_unsafe:
                logger.info(
                    "Gemma4 vLLM: prefill device trace for {} runs hybrid per-layer "
                    "page tables — warmup pre-captures each bucket through "
                    "prefill_forward (per-layer routing active), so runtime replays "
                    "instead of cold-eager capturing (#49083 fix). prefill_trace_enabled={}.",
                    model_path,
                    prefill_trace_enabled,
                )
            _patch_model_args(
                model_args_i,
                submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                model_path=model_path,
                prefill_trace_enabled=prefill_trace_enabled,
                bounded_sliding=bounded_sliding_kv_cache,
            )
            # The shared TT vLLM cache allocator reads ``model.args.optimizations``;
            # mirror the text-transformer wrappers by exposing model_args here.
            model_i.args = model_args_i
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    def _chunk_prefill_page_table(self, page_table, *, user_id, model_id=-1, kv_cache=None):
        """Use a full-attention layer's per-layer table for multi-chunk fill.

        vLLM's legacy ``page_table`` is ``block_tables_per_group[0]`` — the
        sliding group. Full-attention ``paged_fill_cache`` writes via
        ``chunk_page_table``, which must be sliced from the *full* group's
        block IDs at that group's **effective** block_size (the same value
        ``paged_fill_cache`` / chunked SDPA use). On 31B TP=4 that is 128
        (HMA-shared buffer declared as sliding ``[4, 64, 256]`` → full view
        ``eff_bs = 4*64*256/(1*512) = 128``), matching vLLM's unified full-
        group page-table column stride. Using head_dim-only scaling (32)
        walks past the allocated columns and fills chunk 2+ from zeros —
        the 16k garbage cliff. Sliding fill still uses ``layer_page_table``
        (``cache_position_modulo`` path in ``attention/prefill.py``).
        """
        del user_id
        from models.demos.gemma4.tt.attention.operations import effective_block_size

        model = self.model[model_id]
        per_layer = getattr(model, "_active_page_tables_per_layer", None)
        if per_layer is None or kv_cache is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        text_config = getattr(model.hf_config, "text_config", model.hf_config)
        layer_types = getattr(text_config, "layer_types", None) or []
        full_idx = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), None)
        if full_idx is None or full_idx >= len(per_layer) or per_layer[full_idx] is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        full_pt = per_layer[full_idx]
        # Persistent / ttnn tensors are device-side; chunk slicing needs a host
        # torch table. Fall back to legacy if the stash was already converted.
        if not isinstance(full_pt, torch.Tensor):
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        if full_idx >= len(kv_cache) or kv_cache[full_idx] is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        # Sequential path: legacy page_table is already the 1-row slice but
        # ``user_id`` is forced to 0 — pick the matching full-attn row.
        if (
            isinstance(page_table, torch.Tensor)
            and page_table.dim() > 1
            and int(page_table.shape[0]) == 1
            and int(full_pt.shape[0]) > 1
        ):
            row = self._match_page_table_row(page_table, per_layer)
            if row is not None:
                full_pt = full_pt[row : row + 1]

        cache = kv_cache[full_idx][0]
        attn = model.layers[full_idx].self_attn
        cfg = attn.config
        weights = getattr(attn, "weights", None)
        tp = getattr(getattr(model, "mesh_config", None), "tp", 1) or 1
        if weights is not None and getattr(weights, "kv_replicated", False):
            nkv_local = 1
        else:
            nkv_local = max(1, int(cfg.num_key_value_heads) // tp)
        full_block_size = int(effective_block_size(cache, int(cfg.head_dim), nkv_local))
        return full_pt, full_block_size

    def _prefill_user_chunk_plan(self, tokens, kwargs):
        """Return ``(chunk_users, prefill_seq_len)`` when B must be micro-batched.

        ``gemma4.tt.generator.Generator.prefill_forward_text`` chunks B>4 to
        avoid the P150x8 all_gather hang. The vLLM bridge's MRO skips that
        class — without this plan, the plugin's true-batched B=32 prefill
        wedges after cold capture.
        """
        if tokens is None or tokens.shape[0] <= 1:
            return None, None
        batch_size = int(tokens.shape[0])
        batch_seq_len = int(tokens.shape[1])
        prompt_lens = kwargs.get("prompt_lens")
        start_pos = kwargs.get("start_pos")
        prompt_lens_list = prompt_lens if prompt_lens is not None else [batch_seq_len] * batch_size
        if not isinstance(prompt_lens_list, list):
            prompt_lens_list = prompt_lens_list.tolist()
        num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached)
            for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
        ]
        page_table = kwargs.get("page_table")
        # Same padded bucket is enough: attention/prefill.py caps each slot's
        # paged_fill with valid_seq_lens (from last_token_idx). Hetero actual
        # lengths no longer force sequential.
        can_batch_prefill = (
            page_table is not None
            and batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and all(n == 0 for n in num_cached_per_user)
        )
        if not can_batch_prefill:
            return None, None
        padded_batch = next(
            (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
            self.model_args[0].max_batch_size,
        )
        max_users = resolve_batched_prefill_chunk_users(padded_batch, prefill_seq_lens[0])
        if batch_size <= max_users or padded_batch > self.model_args[0].max_batch_size:
            return None, prefill_seq_lens[0]
        return max_users, prefill_seq_lens[0]

    def _slice_prefill_kwargs(self, kwargs, chunk_start, chunk_end):
        """Slice host prefill tensors for a user micro-batch.

        Remapped sliding block IDs are already global (from the full-batch
        remap). Re-slot users to ``0..chunk_size-1`` so tt_transformers'
        padded_batch path stays in-range; physical block IDs stay correct.
        """
        chunk_size = chunk_end - chunk_start
        chunk = dict(kwargs)
        tokens = kwargs["tokens"]
        chunk["tokens"] = tokens[chunk_start:chunk_end]
        page_table = kwargs.get("page_table")
        if page_table is not None:
            chunk["page_table"] = page_table[chunk_start:chunk_end]
        prompt_lens = kwargs.get("prompt_lens")
        if prompt_lens is not None:
            if isinstance(prompt_lens, torch.Tensor):
                chunk["prompt_lens"] = prompt_lens[chunk_start:chunk_end]
            else:
                chunk["prompt_lens"] = list(prompt_lens)[chunk_start:chunk_end]
        start_pos = kwargs.get("start_pos")
        if start_pos is not None:
            chunk["start_pos"] = list(start_pos)[chunk_start:chunk_end]
        # Local slots 0..N-1 (required by padded_batch placement).
        chunk["empty_slots"] = list(range(chunk_size))
        return chunk

    def _merge_prefill_chunk_results(self, batch_size, sampling_params, chunk_results):
        merged_output = None
        merged_tokens = None
        merged_log_probs = None
        for chunk_start, chunk_end, chunk_result in chunk_results:
            if sampling_params is not None:
                chunk_tokens, chunk_log_probs = chunk_result
                if merged_tokens is None:
                    merged_tokens = torch.zeros(
                        (batch_size, *chunk_tokens.shape[1:]),
                        dtype=chunk_tokens.dtype,
                        device=chunk_tokens.device,
                    )
                merged_tokens[chunk_start:chunk_end] = chunk_tokens
                if isinstance(chunk_log_probs, tuple):
                    if merged_log_probs is None:
                        merged_log_probs = (
                            torch.zeros(
                                (batch_size, *chunk_log_probs[0].shape[1:]),
                                dtype=chunk_log_probs[0].dtype,
                                device=chunk_log_probs[0].device,
                            ),
                            torch.zeros(
                                (batch_size, *chunk_log_probs[1].shape[1:]),
                                dtype=chunk_log_probs[1].dtype,
                                device=chunk_log_probs[1].device,
                            ),
                        )
                    merged_log_probs[0][chunk_start:chunk_end] = chunk_log_probs[0]
                    merged_log_probs[1][chunk_start:chunk_end] = chunk_log_probs[1]
                else:
                    if merged_log_probs is None:
                        merged_log_probs = torch.zeros(
                            (batch_size, *chunk_log_probs.shape[1:]),
                            dtype=chunk_log_probs.dtype,
                            device=chunk_log_probs.device,
                        )
                    merged_log_probs[chunk_start:chunk_end] = chunk_log_probs
            else:
                if merged_output is None:
                    merged_output = torch.zeros(
                        (batch_size, *chunk_result.shape[1:]),
                        dtype=chunk_result.dtype,
                        device=chunk_result.device,
                    )
                merged_output[chunk_start:chunk_end] = chunk_result
        if sampling_params is not None:
            return merged_tokens, merged_log_probs
        return merged_output

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
            kwargs["tokens"] = tokens

        chunk_users, prefill_seq_len = self._prefill_user_chunk_plan(tokens, kwargs)
        enable_trace = kwargs.pop("enable_trace", True)
        batch_size = int(tokens.shape[0]) if tokens is not None else 1
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)

        prompt_lens = kwargs.get("prompt_lens")
        seq_len = None
        if prompt_lens is not None:
            try:
                seq_len = int(max(prompt_lens))
            except (TypeError, ValueError):
                seq_len = None
        if seq_len is None and tokens is not None and hasattr(tokens, "shape"):
            seq_len = int(tokens.shape[-1])

        # Remap the *full* batch first so sliding block IDs stay global. Chunk
        # loops only slice those tables (never re-remap local rows 0..N).
        full_page_tables = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        full_page_tables = self._pad_sliding_page_tables_for_bounded(full_page_tables, kwargs.get("kv_cache"))
        full_page_tables = self._pad_page_tables_batch_to_max(full_page_tables)
        if self._bounded_sliding_kv_cache and full_page_tables:
            sliding_idxs = self._sliding_layer_indices()
            if sliding_idxs and full_page_tables[sliding_idxs[0]] is not None:
                kwargs["page_table"] = full_page_tables[sliding_idxs[0]]
            # Bounded rings reuse dense physical blocks [u*W,(u+1)*W) across
            # requests. Clear only on a fresh prefill (all start_pos == 0).
            # start_pos may be a list or numpy/torch vector — never use
            # ``array or []`` (ambiguous truth value for multi-element arrays).
            start_pos_for_clear = kwargs.get("start_pos")
            if start_pos_for_clear is None:
                self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"))
            else:
                try:
                    start_vals = [int(p) for p in list(start_pos_for_clear)]
                except TypeError:
                    start_vals = [int(start_pos_for_clear)]
                if all(p == 0 for p in start_vals):
                    self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"))

        # Align vLLM chunked-prefill continuations to SDPA q_chunk_size (128).
        # tokens[:, :prompt_lens] still holds the full prefix, so aligning
        # start_pos down re-prefills the unaligned boundary (Galaxy pattern).
        start_pos = kwargs.get("start_pos")
        if start_pos is not None:
            kwargs["start_pos"] = align_num_cached_tokens_to_sdpa([int(n) for n in start_pos])

        t0 = time.perf_counter()

        # B>4 true-batched prefill hangs on P150x8 after the first all_gather.
        # Micro-batching with remapped local slots (0..chunk) also breaks decode:
        # KV lands in the right physical blocks but per-slot decode state does not.
        # Force the proven per-user prefill loop (global empty_slots) instead.
        force_sequential = chunk_users is not None
        if force_sequential:
            logger.info(
                "Gemma4 vLLM: sequential prefill for batch_size={} " "(true-batched B>{} hangs on P150x8; user_cap={})",
                batch_size,
                max_batched_prefill_users(),
                max_batched_prefill_users(),
            )

        # Decide if this call will truly batch (same *padded* bucket; hetero
        # actual OK via per-slot valid_seq_lens). Sequential keeps per-layer
        # tables and slices to the active row (see mixin
        # ``_activate_sequential_per_layer_row``).
        prompt_lens_list = prompt_lens
        if prompt_lens_list is not None and not isinstance(prompt_lens_list, list):
            prompt_lens_list = list(prompt_lens_list)
        start_pos_for_plan = kwargs.get("start_pos")
        num_cached_for_plan = (
            [int(n) for n in start_pos_for_plan]
            if start_pos_for_plan is not None
            else ([0] * len(prompt_lens_list) if prompt_lens_list is not None else [0])
        )
        if prompt_lens_list is not None:
            prefill_seq_lens_plan = [
                get_padded_prefill_len(int(seq_len) - num_cached)
                for seq_len, num_cached in zip(prompt_lens_list, num_cached_for_plan)
            ]
            padded_lens_equal = len(set(prefill_seq_lens_plan)) == 1
        else:
            prefill_seq_lens_plan = None
            padded_lens_equal = True
        will_batch = (
            batch_size > 1
            and not force_sequential
            and kwargs.get("page_table") is not None
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and padded_lens_equal
            and all(n == 0 for n in num_cached_for_plan)
        )
        use_sequential = batch_size > 1 and not will_batch

        # Always install per-layer tables when available. Under bounded sliding
        # kwargs["page_table"] is the remapped *sliding* table; clearing the
        # per-layer stash makes full-attention layers inherit it (empty thought
        # / ~10% GPQA). Sequential tt_transformers still slices a 1-row legacy
        # page_table and forces user_id=0 — Gemma4Model slices the multi-row
        # per-layer stash down to that active row (see ttnn_prefill_forward).
        per_submesh = self._chunk_page_tables_per_dp(full_page_tables)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        else:
            for m in self.model:
                if hasattr(m, "_active_page_tables_per_layer"):
                    del m._active_page_tables_per_layer
        if use_sequential:
            logger.info(
                "Gemma4 vLLM: sequential per-user prefill for batch_size={} "
                "(per-layer page tables kept for hybrid/bounded full-attn)",
                batch_size,
            )

        if prefill_seq_len is not None:
            prefill_seq_lens = [prefill_seq_len]
        elif prefill_seq_lens_plan is not None:
            prefill_seq_lens = prefill_seq_lens_plan
        elif seq_len is not None:
            prefill_seq_lens = [get_padded_prefill_len(seq_len)]
        elif tokens is not None:
            prefill_seq_lens = [get_padded_prefill_len(int(tokens.shape[1]))]
        else:
            prefill_seq_lens = [128]
        can_batch = will_batch
        enable_trace = resolve_gemma4_prefill_trace_enable(
            enable_trace,
            self.model[0],
            self.model_args[0],
            batch_size=1 if use_sequential else batch_size,
            prefill_seq_lens=prefill_seq_lens,
            can_batch_prefill=can_batch,
        )
        kwargs["enable_trace"] = enable_trace

        args0 = self.model_args[0]
        prev_disable = getattr(args0, "disable_batched_prefill", False)
        if use_sequential:
            args0.disable_batched_prefill = True
        try:
            with self._route_per_layer_page_tables(per_submesh):
                out = super().prefill_forward_text(**kwargs)
        finally:
            if use_sequential:
                args0.disable_batched_prefill = prev_disable
            self._clear_sequential_batch_page_tables()

        # Device work is synchronous after the TT forward returns — same
        # wall clock the metal demo attributes to ``inference_prefill`` / TTFT.
        dt = time.perf_counter() - t0
        ttft_ms = dt * 1000.0
        prefill_tok_s = (float(seq_len) / dt) if (seq_len and dt > 0) else 0.0
        chunk = getattr(self.model_args[0], "max_prefill_chunk_size", None)
        logger.info(
            "[gemma4-vllm-perf] prefill TTFT={:.1f} ms | prefill_tok/s={:.2f} | "
            "seq_len={} | max_prefill_chunk_size={} | bounded_sliding={} | batch={}",
            ttft_ms,
            prefill_tok_s,
            seq_len,
            chunk,
            self._bounded_sliding_kv_cache,
            batch_size,
        )
        # Reset decode accumulators at the start of each new generate.
        self._perf_decode_tokens = 0
        self._perf_decode_s = 0.0
        return out

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        page_tables_per_layer = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        page_tables_per_layer = self._pad_sliding_page_tables_for_bounded(page_tables_per_layer, kwargs.get("kv_cache"))
        # Do *not* pad decode page tables to max_batch — keep the plugin's
        # nearest-bucket batch so B=1 uses the B=1 decode trace / SDPA grid.
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        # If persistent page-table buffers grew after decode-trace capture,
        # drop the stale Metal traces so the next step recaptures against
        # the new addresses (see Gemma4Model._page_tables_to_ttnn).
        # Under async decode this must not happen after warmup — a grow while
        # a prior step is still in flight rebinds buffers the pending read
        # still references (#51186).
        if any(getattr(m, "_invalidate_decode_traces_after_page_table_realloc", False) for m in self.model):
            after_warmup = bool(getattr(self, "_decode_warmup_complete", False))
            self.trace_ids_decode = defaultdict(lambda: None)
            self.trace_inputs_decode = defaultdict(lambda: None)
            self.trace_output_decode = defaultdict(lambda: None)
            self._prev_decode_batch = None
            for m in self.model:
                m._invalidate_decode_traces_after_page_table_realloc = False
            log = logger.error if after_warmup else logger.warning
            suffix = " [AFTER WARMUP — unsafe under async_scheduling]" if after_warmup else ""
            log(
                "Gemma4 vLLM: cleared decode traces after per-layer page-table "
                f"buffer grow (addresses changed){suffix}"
            )
        t0 = time.perf_counter()
        with self._route_per_layer_page_tables(per_submesh):
            # Route through ``ChunkedPrefillPageTableGuardMixin.decode_forward``
            # (Gemma4-safe async-ahead merge). Do not call
            # ``super(HybridAttentionForCausalLM, ...)`` — that skips the mixin
            # and hits shared ``Generator.decode_forward`` (OOB slot_remap /
            # bucket IndexError under concurrent vLLM). Also avoid plain
            # ``HybridAttentionForCausalLM.decode_forward`` (NotImplementedError).
            out = super().decode_forward(*args, **kwargs)
        self._perf_decode_tokens += 1
        do_log = self._perf_decode_tokens % self._perf_log_every == 0
        # Sync only when measuring (or GEMMA4_VLLM_DECODE_SYNC_EVERY=1); token
        # readback already synchronizes for correctness.
        sync_mode = self._perf_decode_sync_every
        should_sync = sync_mode == "1" or sync_mode == "always" or (sync_mode == "log" and do_log)
        if should_sync:
            try:
                mesh = getattr(self.model_args[0], "mesh_device", None)
                if mesh is not None:
                    ttnn.synchronize_device(mesh)
            except Exception:
                pass
        dt = time.perf_counter() - t0
        self._perf_decode_s += dt
        if do_log:
            tok_s_u = self._perf_decode_tokens / self._perf_decode_s if self._perf_decode_s > 0 else 0.0
            ms_tok = (self._perf_decode_s / self._perf_decode_tokens) * 1000.0 if self._perf_decode_tokens else 0.0
            logger.info(
                "[gemma4-vllm-perf] decode tok/s/user={:.2f} | ms/token={:.2f} | "
                "tokens={} | bounded_sliding={} | decode_batch={}",
                tok_s_u,
                ms_tok,
                self._perf_decode_tokens,
                self._bounded_sliding_kv_cache,
                getattr(self, "_prev_decode_batch", None),
            )
        return out

    def allocate_kv_cache(self, *args, **kwargs):
        # Legacy uniform path (vLLM falls back here when ``get_kv_cache_spec``
        # isn't consulted). The hybrid path uses ``allocate_kv_cache_per_layer``
        # inherited from :class:`HybridAttentionForCausalLM`.
        return allocate_vllm_kv_cache(
            *args,
            **kwargs,
            dp_model=self.model,
            tt_cache_path=self.cache_path,
        )

    def _text_config(self):
        model = self.model[0]
        return getattr(model.hf_config, "text_config", model.hf_config)

    def _sliding_layer_indices(self) -> list[int]:
        layer_types = list(getattr(self._text_config(), "layer_types", None) or [])
        return [i for i, lt in enumerate(layer_types) if lt == "sliding_attention"]

    def _bounded_sliding_physical_blocks(self, block_size: int) -> int | None:
        """Demo-parity sliding pool size: ``(sliding_window/block_size) * B``.

        Used when ``bounded_sliding`` is on so hybrid-OFF UniformType specs
        (full ``num_blocks`` for every layer) do not allocate ~256k-length
        sliding buffers and OOM at long ISL.
        """
        if not self._bounded_sliding_kv_cache or block_size <= 0:
            return None
        sliding_window = getattr(self._text_config(), "sliding_window", None)
        if sliding_window is None or int(sliding_window) % block_size != 0:
            return None
        max_batch = int(self.model_args[0].max_batch_size)
        return (int(sliding_window) // block_size) * max_batch

    def _clear_bounded_sliding_kv_rings(self, kv_cache) -> None:
        """Zero sliding-layer paged KV buffers before a fresh prefill.

        Bounded mode remaps every user onto a fixed physical ring; those
        buffers are not freshly allocated per request. Stale contents from
        warmup or a prior generate corrupt short-prompt next-token logits
        (notably closing the gemma4 thought channel immediately).
        """
        if not self._bounded_sliding_kv_cache or kv_cache is None:
            return
        sliding_idxs = set(self._sliding_layer_indices())
        if not sliding_idxs:
            return
        # kv_cache: [submesh][layer] -> [k, v] (or layer -> [k, v] when undped).
        submeshes = kv_cache if isinstance(kv_cache, (list, tuple)) else [kv_cache]
        cleared = 0
        for sub in submeshes:
            if sub is None:
                continue
            layers = sub if isinstance(sub, (list, tuple)) else [sub]
            for li, layer_kv in enumerate(layers):
                if li not in sliding_idxs or layer_kv is None:
                    continue
                pair = layer_kv if isinstance(layer_kv, (list, tuple)) else (layer_kv,)
                for cache_t in pair:
                    if cache_t is None:
                        continue
                    try:
                        z = ttnn.zeros_like(cache_t)
                        ttnn.copy(z, cache_t)
                        z.deallocate(True)
                        cleared += 1
                    except Exception as e:
                        logger.warning(
                            "Gemma4 vLLM: failed to clear bounded sliding KV "
                            "(layer={}, err={}) — stale ring may remain",
                            li,
                            e,
                        )
        if cleared:
            logger.info(
                "Gemma4 vLLM: cleared {} bounded sliding KV buffers before prefill",
                cleared,
            )

    def _shrink_bounded_sliding_kv_specs(self, per_layer_specs):
        """Rewrite sliding-layer ``num_blocks`` to the bounded physical pool."""
        sliding_idxs = set(self._sliding_layer_indices())
        if not sliding_idxs:
            return per_layer_specs
        # Specs are (shape, dtype, tensor_idx); shape[0]=num_blocks, shape[2]=block_size.
        sample_bs = None
        for i, (shape, _, _) in enumerate(per_layer_specs):
            if i in sliding_idxs:
                sample_bs = int(shape[2])
                break
        if sample_bs is None:
            return per_layer_specs
        new_blocks = self._bounded_sliding_physical_blocks(sample_bs)
        if new_blocks is None:
            return per_layer_specs
        out = []
        shrunk = 0
        for i, (shape, dtype, tensor_idx) in enumerate(per_layer_specs):
            if i in sliding_idxs and int(shape[0]) > new_blocks:
                shape = (new_blocks, *shape[1:])
                shrunk += 1
            out.append((shape, dtype, tensor_idx))
        if shrunk:
            logger.info(
                "Gemma4 vLLM: bounded sliding — sized {}/{} sliding KV buffers "
                "to {} blocks (sliding_window/block_size * max_batch={}), "
                "matching metal demo (avoids full-ISL sliding DRAM OOM).",
                shrunk,
                len(sliding_idxs),
                new_blocks,
                int(self.model_args[0].max_batch_size),
            )
        return out

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate per-layer KV cache, then alias KV-shared layers to
        their source layer's buffer.

        Gemma4-E2B / -E4B have a Gemma3n-style "num_kv_shared_layers"
        optimization where the last N layers reuse an earlier layer's
        K/V instead of computing+storing their own. The model side
        encodes this via ``self.kv_shared_layer_map`` (layer_idx →
        source_idx) and ``attention/{prefill,decode}.py`` skips
        ``paged_{fill,update}_cache`` whenever a layer is flagged as
        shared. vLLM's hybrid kv-cache manager is unaware of this
        TT-specific reuse and allocates a distinct buffer for every
        layer; without the post-allocator alias the shared layers'
        SDPA reads land on zero-initialized buffers.

        Important: aliasing the buffer is *necessary but not sufficient*.
        Source and shared layers share an attention *type* (sliding or
        full), but vLLM's hybrid manager constructs more groups than
        just "one per type" — for Gemma4-E2B with 35 layers in the
        4-sliding-then-1-full pattern, vLLM produces 5 groups of 7
        layers each (4 sliding sub-groups + 1 full group), and each
        physical tensor is shared by one layer from each group. That
        means layer 13 (sliding, in group[3]) and layer 15 (sliding,
        in group[0]) have *different* per-layer page_tables, so
        aliasing only the buffer leaves layer 15 reading the wrong
        slot of the shared tensor — whatever group[0]'s layer 10 wrote
        there, not what layer 13 wrote. The buffer alias must be
        paired with a per-layer-page-table alias in
        :meth:`_block_tables_per_layer_with_kv_share` so the shared
        layer indexes the buffer the same way the source did.

        When ``bounded_sliding`` is on (hybrid groups still OFF), sliding
        layer buffers are also shrunk to the demo window pool before
        allocation — see :meth:`_shrink_bounded_sliding_kv_specs`.
        """
        per_layer_specs = self._shrink_bounded_sliding_kv_specs(per_layer_specs)
        kv_cache = super().allocate_kv_cache_per_layer(per_layer_specs)
        for submesh_idx, submesh_kv in enumerate(kv_cache):
            kv_shared_map = getattr(self.model[submesh_idx], "kv_shared_layer_map", None)
            if not kv_shared_map:
                continue
            for layer_idx, source_idx in kv_shared_map.items():
                submesh_kv[layer_idx] = submesh_kv[source_idx]
        return kv_cache

    def _ensure_page_tables_per_layer(self, page_tables_per_layer, page_table):
        """Broadcast legacy ``page_table`` to per-layer whenever the plugin
        only sent the legacy view.

        Parent only broadcasts for hybrid-ON. Gemma4 always routes fills
        through ``_chunk_prefill_page_table`` / per-layer aliases (incl.
        unbounded hybrid-OFF + vLLM APC), so a missing stash falls back to a
        truncated legacy table and continuation chunks write through -1
        (#51186). Broadcasting the (full-width) legacy map is cheap and keeps
        APC absolute fills correct.
        """
        if page_tables_per_layer is not None or page_table is None:
            return page_tables_per_layer
        num_layers = len(self.model[0].layers)
        return [page_table] * num_layers

    def _build_per_layer_page_tables(self, page_tables_per_layer, legacy_page_table):
        """Compose the inherited per-layer broadcast/passthrough with
        the Gemma4-specific kv-share alias.

        Composition logic, kept in one place so
        :meth:`prefill_forward` / :meth:`decode_forward` each take one
        call instead of remembering to chain two helpers:

        1. :meth:`_ensure_page_tables_per_layer`
           — broadcast a legacy single ``page_table`` to per-layer when
           the plugin only sent the legacy view (warmup, tests).
        2. :meth:`_apply_kv_share_to_per_layer_page_tables` — for every
           ``(shared_idx, source_idx)`` in the model's
           ``kv_shared_layer_map``, re-point the shared layer's
           page_table at the source's. See the
           :meth:`allocate_kv_cache_per_layer` docstring for why
           aliasing the buffer alone leaves the shared layer reading
           a different layer's slot of the shared HMA tensor.
        """
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, legacy_page_table)
        return self._apply_kv_share_to_per_layer_page_tables(page_tables_per_layer)

    def _apply_kv_share_to_per_layer_page_tables(self, page_tables_per_layer):
        """Replace every kv-shared layer's per-layer page_table with
        its source layer's per-layer page_table.

        The buffer alias in :meth:`allocate_kv_cache_per_layer` makes
        ``caches[shared] is caches[source]``; this method makes
        ``page_tables[shared] is page_tables[source]``. Together they
        ensure the shared layer reads exactly the (buffer, block IDs)
        the source layer wrote — without this, the shared layer reads
        the slot in the HMA-shared buffer that the layer in its own
        kv-cache sub-group wrote, which is some other layer's K/V.
        See [[gemma4-kv-share-page-table-alias]] for the diagnosis
        path.
        """
        if not page_tables_per_layer:
            return page_tables_per_layer
        kv_shared_map = getattr(self.model[0], "kv_shared_layer_map", None) or {}
        if not kv_shared_map:
            return page_tables_per_layer
        out = list(page_tables_per_layer)
        for layer_idx, source_idx in kv_shared_map.items():
            if 0 <= layer_idx < len(out) and 0 <= source_idx < len(out):
                out[layer_idx] = out[source_idx]
        return out

    def _pad_page_tables_batch_to_max(self, page_tables_per_layer):
        """Pad host page-table batch dim up to ``max_batch_size``.

        Decode warmup captures Metal traces against persistent buffers sized
        at max batch. Prefill often passes B=1 or B=31; padding here makes
        the first allocation (and every subsequent copy) match that width so
        we never grow/orphan trace addresses. Unused rows are filled with 0
        (vLLM null block).
        """
        if not page_tables_per_layer:
            return page_tables_per_layer
        max_b = int(self.model_args[0].max_batch_size)
        out = []
        for pt in page_tables_per_layer:
            if pt is None or not isinstance(pt, torch.Tensor):
                out.append(pt)
                continue
            pt2 = pt if pt.dim() > 1 else pt.unsqueeze(0)
            if int(pt2.shape[0]) >= max_b:
                out.append(pt2)
                continue
            padded = torch.zeros((max_b, int(pt2.shape[1])), dtype=torch.int32)
            padded[: pt2.shape[0], :] = pt2.to(dtype=torch.int32)
            out.append(padded)
        return out

    def _pad_sliding_page_tables_for_bounded(self, page_tables_per_layer, kv_cache):
        """Remap sliding-layer page tables onto the bounded physical pool.

        With hybrid groups OFF, vLLM hands every layer the same full-ISL
        block table (global IDs into ``num_blocks≈max_model_len/block_size``).
        Bounded mode allocates only ``sliding_window/block_size * B`` physical
        blocks per sliding layer (see :meth:`_shrink_bounded_sliding_kv_specs`),
        so those global IDs would OOB. Rebuild each sliding row with dense
        local IDs — same layout as ``build_hybrid_page_tables`` in the metal
        demo: user ``u`` owns ``[u*W, (u+1)*W)`` where ``W=sliding_window/block_size``.

        Tables are sized to exactly ``W`` columns so ``cache_position_modulo``
        shape checks pass on short prompts without retaining vLLM's full-ISL
        width (unused under modulo wrap).

        Full-attention layers are left alone.
        """
        if not self._bounded_sliding_kv_cache:
            return page_tables_per_layer
        if not page_tables_per_layer:
            return page_tables_per_layer
        model = self.model[0]
        text_config = self._text_config()
        sliding_window = getattr(text_config, "sliding_window", None)
        layer_types = getattr(text_config, "layer_types", None)
        if sliding_window is None or layer_types is None:
            return page_tables_per_layer

        # Prefer a *sliding* layer's K-cache block_size (after shrink). Full
        # layers may share the same declared block_size under UniformType.
        block_size = None
        sliding_idxs = self._sliding_layer_indices()
        if kv_cache is not None and sliding_idxs:
            try:
                block_size = int(kv_cache[0][sliding_idxs[0]][0].shape[2])
            except (TypeError, IndexError, AttributeError):
                block_size = None
        if block_size is None and kv_cache is not None:
            try:
                block_size = int(kv_cache[0][0][0].shape[2])
            except (TypeError, IndexError, AttributeError):
                block_size = None
        if block_size is None:
            try:
                block_size = int(model.layers[0].self_attn.kv_cache[0].shape[2])
            except (TypeError, IndexError, AttributeError):
                return page_tables_per_layer

        if sliding_window % block_size != 0:
            return page_tables_per_layer
        target_cols = int(sliding_window) // block_size

        # Dense sliding IDs depend only on (batch, W) — cache across decode
        # steps so we don't rebuild 50 layer tables every token.
        batch = None
        for i, pt in enumerate(page_tables_per_layer):
            if (
                pt is not None
                and i < len(layer_types)
                and layer_types[i] == "sliding_attention"
                and isinstance(pt, torch.Tensor)
            ):
                batch = int(pt.shape[0])
                break
        cache = getattr(self, "_bounded_sliding_pt_cache", None)
        cached_row = None
        if cache is not None and cache[0] == batch and cache[1] == target_cols:
            cached_row = cache[2]

        out = []
        for i, pt in enumerate(page_tables_per_layer):
            if (
                pt is None
                or i >= len(layer_types)
                or layer_types[i] != "sliding_attention"
                or not hasattr(pt, "shape")
                or not isinstance(pt, torch.Tensor)
            ):
                out.append(pt)
                continue
            batch = int(pt.shape[0])
            if cached_row is not None and cached_row.shape[0] == batch:
                out.append(cached_row)
                continue
            # Always W columns (demo layout). Keeping vLLM's full-ISL width
            # here thrash-reallocates persistent buffers vs short prefill
            # tables and is unused under cache_position_modulo.
            remapped = torch.empty((batch, target_cols), dtype=torch.int32)
            for u in range(batch):
                remapped[u] = torch.arange(u * target_cols, (u + 1) * target_cols, dtype=torch.int32)
            self._bounded_sliding_pt_cache = (batch, target_cols, remapped)
            cached_row = remapped
            out.append(remapped)
        return out
