# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import time
from collections import defaultdict

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.dflash_constants import VERIFY_WIDTH_MARGIN
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
    # ``GEMMA4_SUPPORTS_ASYNC_DECODE=0``. PLI models narrow the instance dict
    # in ``__init__``; that does not reach the platform, so PLI still needs
    # the kill-switch to disable async_scheduling.
    @staticmethod
    def _spec_real_batch(tokens, start_pos):
        """Rows that belong to REAL requests, ignoring the runner's padding.

        The runner pads a decode batch up to a wire bucket
        (``decode_pad_to``) and pads the padded rows' positions with -1, by its
        own convention: "Pad positions with -1 to indicate no position". So
        ``tokens.shape[0]`` is the WIRE width, not the number of requests.

        The adaptive gate must use the real count. Reading the wire width made a
        SOLO request whose bucket padded above 1 look batched, so the model
        served it as plain baseline at width 1 while the scheduler -- which
        counts scheduled requests, and was right -- had reserved a full block.
        The commit then died on the width check:

            ValueError: Model output width violates output_tokens_per_step:
            1 != 64

        Seen on a BH Galaxy DP=4 server at 121k ISL, where the smallest decode
        bucket is above 1; a P150x8 server pads solo decodes to 1 and never hit
        it.
        """
        if start_pos is None:
            return int(tokens.shape[0])
        try:
            return max(1, int((start_pos.reshape(-1) >= 0).sum()))
        except Exception:
            return int(tokens.shape[0])

    @staticmethod
    def _spec_pt_identity(page_table):
        """Stable identity for the request a spec session belongs to: the first
        block id of its page-table row.

        Serving runs with prefix caching OFF, so live requests own disjoint KV
        blocks and a row's first block does not move while the request lives.
        Available on BOTH the prefill and the decode call (``page_table``
        kwarg), which is what lets the owner recorded at capture time be
        re-checked before the taps are bootstrapped.
        """
        if page_table is None:
            return None
        try:
            row = page_table[0] if page_table.dim() > 1 else page_table
            return int(row.reshape(-1)[0])
        except Exception:
            return None

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
        # would restage a stale token. Narrow the instance dict so runtime
        # readers (the decode-sync default below) see False. The vLLM TT plugin
        # snapshots the class attribute during check_and_update_config, before
        # this runs, so this does not disable scheduler_config.async_scheduling;
        # set GEMMA4_SUPPORTS_ASYNC_DECODE=0 for that.
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
        # ``self.sampling = None``. Narrow the instance dict so runtime readers
        # see False. The platform already snapshot the class attribute, so this
        # does not revise sample_on_device_mode.
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
        # Snapshot ring occupancy *before* the remap: the remap inserts the
        # request being prefilled into the slot map, so a check made after it
        # would see a non-empty map even for the very first request and skip
        # the clear it actually needs (stale warmup KV -> garbage for exactly
        # that one user).
        rings_live_before_prefill = len(getattr(self, "_bounded_ring_slot_map", None) or {})
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
            if not rings_live_before_prefill:
                # Fresh wave (no live rings): drop captured decode traces so the
                # next decode recaptures against this wave's state. Replaying a
                # decode trace captured before the wave -- in practice the
                # warmup-captured traces -- corrupts every batch after the first
                # even when all trace input tensors are restaged each step
                # (measured on P150x8 / 12B / 32k bounded: batch 1 clean, every
                # later batch nondeterministic garbage at any concurrency >= 2;
                # ring fingerprints show prefill hidden states already diverging
                # for identical inputs). With this release the first serving
                # wave recaptures once and 16+ consecutive batches across
                # concurrencies 1..32 and bucket switches replay it cleanly.
                # Program caches persist, so the one recapture costs seconds.
                self._release_decode_traces_for_fresh_wave()
            start_pos_for_clear = kwargs.get("start_pos")
            if start_pos_for_clear is None:
                self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"), live_before=rings_live_before_prefill)
            else:
                try:
                    start_vals = [int(p) for p in list(start_pos_for_clear)]
                except TypeError:
                    start_vals = [int(start_pos_for_clear)]
                if all(p == 0 for p in start_vals):
                    self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"), live_before=rings_live_before_prefill)

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
            # Batched prefill is opt-in via G4_FORCE_BATCH_PREFILL=1: it works
            # under BOTH unbounded and bounded sliding since the 2026-09 fixes
            # (identity-gated, byte-equal vs the sequential/B=1 references):
            #  - per-layer page tables scattered local->slot rows (below) — the
            #    non-identity empty_slots KV misplacement, debt #1 bug A;
            #  - boot-time batched-shape warmup captures (generator_trace) — the
            #    runtime cold-capture replay corruption, debt #1 bug B;
            #  - bounded per-slot ring fills deferred after lm_head, full-length
            #    under modulo, wrap-boundary merged (attention/prefill.py).
            # Historic exclusion rationales ("per-slot decode state", "deferred
            # lm-head extraction") are retired — the extraction was always
            # correct. Default-off pending the open conc32 small-ISL
            # wave-boundary corruption (~1-3/32 after a prior wave, prefill-mode
            # independent; repro: wave_smoke) and per-quadrant memory budgets
            # (see GEMMA4_MAX_BATCHED_PREFILL_USERS / GEMMA4_TAIL_POOL_SLOTS).
            and os.environ.get("G4_FORCE_BATCH_PREFILL", "0") == "1"
        )
        use_sequential = batch_size > 1 and not will_batch

        if will_batch:
            slots = kwargs.get("empty_slots")
            slots = [int(s) for s in slots] if slots is not None else None
            if slots is not None and slots != list(range(len(slots))):
                # Batched prefill places tokens at their device slots
                # (``prefill_ids[slot] ← user i``) and the per-layer KV fill
                # indexes tables by slot (``paged_fill_cache(batch_idx=slot)``),
                # but the plugin's per-layer tables arrive in *local* prefill
                # order (row i = user i). With non-identity ``empty_slots`` —
                # exactly when another request is mid-decode — user i's KV was
                # written through table row ``slot_i`` of a local-ordered
                # table: each user's KV landed in the next user's blocks and
                # the first user's blocks were never written (debt #1's single
                # corrupted user). Scatter local rows to slot rows, mirroring
                # ``_get_prefill_user_page_table``'s legacy scatter.
                scattered = []
                for pt in full_page_tables or []:
                    if pt is None or not isinstance(pt, torch.Tensor) or pt.dim() < 2:
                        scattered.append(pt)
                        continue
                    out = torch.zeros_like(pt)
                    for i, slot in enumerate(slots):
                        if i < pt.shape[0] and slot < out.shape[0]:
                            out[slot] = pt[i]
                    scattered.append(out)
                full_page_tables = scattered

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
        # Arm the gemma4-owned batched-consumption trace hygiene (per-slot slice
        # dealloc + RM return + retire list in process_logits_after_prefill_trace)
        # only when this call truly batches; the single-user path must keep the
        # original contract (persistent trace output input, TILE return).
        for m in self.model:
            m._g4_batched_prefill_consumption = will_batch

        try:
            with self._route_per_layer_page_tables(per_submesh):
                out = super().prefill_forward_text(**kwargs)
        finally:
            if use_sequential:
                args0.disable_batched_prefill = prev_disable
            for m in self.model:
                m._g4_batched_prefill_consumption = False
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
        # Free tensors retired by the last batched-prefill consumption BEFORE
        # this step's trace replay (the retired list is host-consumed already;
        # leaving the final entry alive across the replay violates the
        # trace-safety contract in process_logits_after_prefill_trace).
        for m in self.model:
            scavenge = getattr(m, "_g4_retire_scavenge", None)
            if scavenge is not None:
                scavenge()
        page_tables_per_layer = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        page_tables_per_layer = self._pad_sliding_page_tables_for_bounded(
            page_tables_per_layer, kwargs.get("kv_cache"), authoritative=True
        )
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
        """Sliding pool size: ``(ring/block_size) * B``.

        Used when ``bounded_sliding`` is on so hybrid-OFF UniformType specs
        (full ``num_blocks`` for every layer) do not allocate ~256k-length
        sliding buffers and OOM at long ISL.

        Sized from the RING, not the bare ``sliding_window``. The ring is what
        positions wrap into (``bounded_ring_modulo``), and on the speculative
        path it is larger than the window so candidate writes at p+1..p+K land
        outside the live window instead of evicting history a candidate query
        still needs. Allocating from the window while wrapping modulo the ring
        writes past the pool, which is why setting the headroom env alone was
        never enough -- allocation, page tables and modulo have to move
        together.
        """
        if not self._bounded_sliding_kv_cache or block_size <= 0:
            return None
        sliding_window = getattr(self._text_config(), "sliding_window", None)
        if sliding_window is None:
            return None
        from models.demos.gemma4.tt.attention import bounded_ring_modulo

        ring = bounded_ring_modulo(int(sliding_window))
        if ring is None or int(ring) % block_size != 0:
            return None
        max_batch = int(self.model_args[0].max_batch_size)
        return (int(ring) // block_size) * max_batch

    def _release_decode_traces_for_fresh_wave(self) -> None:
        """Release captured decode traces (see fresh-wave comment at call site)."""
        released = 0
        try:
            for _key, tids in list(self.trace_ids_decode.items()):
                if not tids:
                    continue
                for mid, tid in tids.items():
                    ttnn.release_trace(self.model_args[mid].mesh_device, tid)
                    released += 1
        except Exception as exc:
            logger.warning("Gemma4 bounded: decode trace release failed: {}", exc)
        if released:
            self.trace_ids_decode = defaultdict(lambda: None)
            self.trace_inputs_decode = defaultdict(lambda: None)
            self.trace_output_decode = defaultdict(lambda: None)
            self._prev_decode_batch = None
            self.prev_page_table = None
            logger.info("Gemma4 bounded: released {} decode trace(s) at fresh wave", released)

    def _clear_bounded_sliding_kv_rings(self, kv_cache, live_before: int = 0) -> None:
        """Zero sliding-layer paged KV buffers before a fresh prefill.

        Bounded mode remaps every user onto a fixed physical ring; those
        buffers are not freshly allocated per request. Stale contents from
        warmup or a prior generate corrupt short-prompt next-token logits
        (notably closing the gemma4 thought channel immediately).

        This zeroes the *whole* sliding pool, i.e. every user's ring. Under
        concurrent serving each arriving request is itself a "fresh prefill"
        (``start_pos == 0``), so doing that unconditionally wipes the KV of
        every request currently decoding — measured as garbage output from
        concurrency 2 upward. ``live_before`` is the ring occupancy sampled
        *before* this prefill's own remap, so the serving path skips the clear
        only when some *other* request still holds a ring; the arriving request
        is handed a ring no live request owns, and SDPA reads only
        ``[cur_pos-W+1, cur_pos]``, which its own prefill has just written, so
        any stale tail left by a finished request is never attended to.
        """
        if not self._bounded_sliding_kv_cache or kv_cache is None:
            return
        if live_before:
            logger.debug(
                "Gemma4 bounded: skipping sliding-ring clear, {} live ring(s) in flight",
                live_before,
            )
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

    def _pad_sliding_page_tables_for_bounded(self, page_tables_per_layer, kv_cache, authoritative=False):
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

        # Ring slots must follow the *request*, not its row in the current
        # page-table tensor. The plugin compacts decode rows onto the occupied
        # requests (``block_tables_for_rows(req_indices, ...)``) and picks the
        # decode bucket from the request *count*, so a live request's row index
        # shifts whenever the running set changes while its KV stays put.
        # Keying dense block IDs on the row index therefore hands a running
        # request a different physical ring between steps and it reads another
        # user's KV (nondeterministic garbage from concurrency 2 upward). Key on
        # vLLM's own global block ID instead: stable for the request's lifetime,
        # and unique while prefix caching is off (Gemma4 declares it off).
        max_slots = int(getattr(self.model_args[0], "max_batch_size", 0) or 0)

        # Derive the identity from a full-attention row when one is available.
        # vLLM substitutes its reserved null block (ID 0) for blocks it has
        # evicted, which for a *sliding* group can zero the very entry used as
        # the key; a full-attention row keeps every block for the request's
        # lifetime. Under hybrid-groups-off both rows are the same table, so
        # this only matters if hybrid groups are turned on later.
        ref = None
        for i, pt in enumerate(page_tables_per_layer):
            if not isinstance(pt, torch.Tensor) or i >= len(layer_types):
                continue
            if layer_types[i] != "sliding_attention":
                ref = pt
                break
            if ref is None:
                ref = pt
        slots_by_row = (
            self._bounded_ring_slots(ref, max_slots or int(ref.shape[0]), authoritative) if ref is not None else None
        )

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
            if slots_by_row is not None and len(slots_by_row) == batch:
                slots = slots_by_row
            else:
                # Per-layer fallback: keys derive from THIS layer's table. Under
                # hybrid-groups-off every layer shares one table so this equals
                # slots_by_row; with hybrid groups on, a sliding row whose block-0
                # was nulled by vLLM could key a different slot than the
                # full-attention reference — if the fallback ever fires alongside
                # a computed reference, surface it instead of diverging silently.
                if slots_by_row is not None and not getattr(self, "_g4_ring_fallback_warned", False):
                    self._g4_ring_fallback_warned = True
                    logger.warning(
                        f"gemma4 bounded ring: layer {i} page table has {batch} rows but the shared "
                        f"slot reference has {len(slots_by_row)} — falling back to per-layer slot "
                        "derivation. Rings can diverge across layers if row identities differ."
                    )
                slots = self._bounded_ring_slots(pt, max_slots or batch, False)
            # Always W columns (demo layout). Keeping vLLM's full-ISL width
            # here thrash-reallocates persistent buffers vs short prefill
            # tables and is unused under cache_position_modulo.
            remapped = torch.zeros((batch, target_cols), dtype=torch.int32)
            for u, slot in enumerate(slots):
                if slot is None:
                    continue  # padded gap row; its position is -1 so it is never read
                remapped[u] = torch.arange(slot * target_cols, (slot + 1) * target_cols, dtype=torch.int32)
            out.append(remapped)
        return out

    @staticmethod
    def _bounded_row_key(row):
        """Stable per-request identity for one page-table row.

        vLLM's global block IDs live as long as the request (Gemma4 runs with
        prefix caching off, so blocks are not shared between requests), which
        makes the first block ID an identity that survives the row index
        moving. An all-zero row is a padded decode gap, not a request.
        """
        if int(row.max()) == 0:
            return None
        return int(row[0])

    def _bounded_ring_slots(self, pt, max_slots, authoritative):
        """Assign each page-table row a persistent bounded-ring slot.

        ``authoritative`` must be set only by the decode path: a decode step
        sees the whole running set, so slots for departed requests can be
        reclaimed there. A prefill call carries just the arriving request, so
        releasing on it would drop every live request's slot and re-hand them
        different rings — the exact corruption this mapping exists to prevent.
        """
        slot_map = getattr(self, "_bounded_ring_slot_map", None)
        if slot_map is None:
            slot_map = {}
            self._bounded_ring_slot_map = slot_map
        keys = [self._bounded_row_key(pt[u]) for u in range(int(pt.shape[0]))]
        # Do NOT release a slot merely because its key is missing from this
        # batch. vLLM does not necessarily schedule every running request in
        # every decode step, so an absent key is not proof the request ended;
        # freeing it there hands a live request's ring to a new arrival. Keep
        # entries until the map is actually full and recycle least-recently-used
        # instead — a request that really finished stops being touched and ages
        # out naturally. ``authoritative`` now only marks a call whose key set
        # is a true running set, which is what refreshes recency.
        if authoritative:
            for k in keys:
                if k is not None and k in slot_map:
                    slot_map[k] = slot_map.pop(k)  # move to end = most recent
            # Remember the last true running set: eviction must prefer keys
            # absent from it — LRU order alone can front a LIVE request that
            # simply was not scheduled in recent decode steps.
            self._g4_last_authoritative_keys = {k for k in keys if k is not None}
        used = set(slot_map.values())
        slots = []
        for k in keys:
            if k is None:
                slots.append(None)
                continue
            slot = slot_map.get(k)
            if slot is None:
                slot = next((s for s in range(max_slots) if s not in used), None)
                if slot is None:
                    # Slots are reclaimed on decode steps, so a batch that
                    # drains completely leaves its keys behind until the next
                    # decode runs. A prefill arriving in that window can find
                    # the map full; the scheduler caps concurrency at
                    # ``max_slots``, so the oldest entry is necessarily a
                    # departed request. Evict it rather than colliding on 0.
                    last_auth = getattr(self, "_g4_last_authoritative_keys", None)
                    victim = next(
                        (vk for vk in slot_map if last_auth is not None and vk not in last_auth),
                        next(iter(slot_map)),
                    )
                    slot = slot_map.pop(victim)
                    logger.warning(
                        "Gemma4 bounded: ring slot map full (max_slots={}); recycling "
                        "least-recently-used key {} to admit a new request",
                        max_slots,
                        victim,
                    )
                slot_map[k] = slot
                used.add(slot)
            slots.append(slot)
        return slots


# ── TT-native speculative serving (B=1, session pattern) ─────────────────────
#
# Selected by the plugin's TT_GEMMA4_SPEC env (arg/gemma4_spec_serving):
# same HF checkpoint/architectures, different decode class. Speculation is
# model-internal (draft + verify inside ONE device step); each decode_forward
# call emits exactly the step's contracted width: the full N-token block on a
# solo decode step (EOS-filled at a genuine stop), or a width-1 row on batched
# decode / prefill-anchor steps (the adaptive scheduler reserves one
# placeholder there). No sentinel padding.
# vLLM's speculative_config stays unset -- the platform assert is untouched.


def dflash_pv_bucket_ladder(max_context, horizon=None, verify=None, max_rungs=None):
    """The ENUMERABLE set of packed-verify buckets a dFlash server can serve.

    ``DFlashFusedDecoder.pv_bucket`` keys the fused trace on
    ``round_up_1024(start + horizon + P_v + 64)``, i.e. on the PROMPT LENGTH, so
    a 256K server has ~258 possible buckets -- far too many to pre-capture, which
    is why capture is per session today. Per-session capture is the part that
    does not conform: section 8 bans a capture that happens during serving, or
    one whose shape can only be known once a request exists. A FIXED SET of
    verify widths chosen at config time is conformant -- every width is known
    before the first request and captured in warmup -- so this ladder is what
    makes the fused verify admissible, not merely a capture-count optimisation.

    The set becomes enumerable because a bucket captured LARGER than a request
    needs is numerically EXACT for it: ``_pv_setup`` caps S_k at capture and
    masks every column past the live top to NEG ("columns past the live top are
    NEG -> exact; the captured program never changes shape"). So a coarse ladder
    plus round-UP serves every prompt length, and the only cost of rounding up
    is wasted verify width.

    The ladder doubles from the smallest useful rung to the largest a request
    can reach, which bounds the waste at <2x the exact bucket while keeping the
    rung count logarithmic in the context (8 rungs at 256K, worst waste 1.50x).

    Pure: config-time arithmetic, no device and no model instance. Returns
    ascending 1024-aligned bucket sizes.
    """
    horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048") if horizon is None else horizon)
    verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5") if verify is None else verify)
    p_v = verify + 1
    tail = horizon + p_v + VERIFY_WIDTH_MARGIN  # what pv_bucket adds on top of ``start``

    def _round(n):
        return ((int(n) + 1023) // 1024) * 1024

    smallest = _round(tail)  # a zero-length prompt still needs the tail
    largest = _round(int(max_context) + tail)
    ladder, rung = [], smallest
    while rung < largest:
        ladder.append(rung)
        rung = _round(rung * 2)
    ladder.append(largest)
    if max_rungs is not None and len(ladder) > int(max_rungs):
        # Keep the largest rungs: dropping a SMALL rung only costs wasted verify
        # width on short prompts, while dropping the largest would leave long
        # prompts with no bucket that fits.
        ladder = ladder[-int(max_rungs) :]
    return ladder


def dflash_bucket_for(start, ladder):
    """Smallest ladder rung that covers ``start``'s exact bucket, or None.

    None means the request is longer than the ladder covers and must fall back
    to a per-session capture.
    """
    horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048"))
    verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))
    need = int(start) + horizon + verify + 1 + VERIFY_WIDTH_MARGIN
    for rung in ladder:
        if rung >= need:
            return rung
    return None


def dflash_width_set(max_context, num_blocks, horizon=None, verify=None, max_rungs=None, block_size=64):
    """The verify WIDTH SET a dFlash server can actually capture.

    ``dflash_pv_bucket_ladder`` gives the widths the CONTEXT needs; this bounds
    them by what the KV POOL can address. The verify page table is
    ``pv_sk // block_size`` entries wide and ``paged_update_cache`` requires
    ``max_num_blocks_per_seq < max_num_blocks``, so a width past the per-request
    block budget kills the engine during warmup:

        max_num_blocks_per_seq must be less than max_num_blocks:
        max_num_blocks_per_seq=4144, max_num_blocks=4128

    which is the ladder's top rung (265216 = round_up_1024(262144 + 2118)) on a
    262144-token server. The top width is therefore the block budget, not the
    rounded-up context, and the last ``P_v + 64`` positions of the context are
    served by it.

    Pure: config-time arithmetic, no device and no model instance.
    """
    ladder = dflash_pv_bucket_ladder(max_context, horizon=horizon, verify=verify, max_rungs=max_rungs)
    cap = ((int(num_blocks) * int(block_size)) // 1024) * 1024 if num_blocks else 0
    if not cap:
        return ladder
    ladder = [w for w in ladder if w <= cap]
    if not ladder or ladder[-1] != cap:
        ladder.append(cap)
    return ladder


def mtp_pv_width_ladder(max_context, draft_len, bucket=1024, max_rungs=None):
    """Verify widths an MTP serving session can migrate through.

    The packed verify's ``S_k`` is fixed per captured trace, and a replay may
    only attend positions the mask it was captured with covers, so a session
    needs ``S_k >= pos + K + 2`` at every position it reaches. Sizing one
    capture to ``anchor_pos + horizon`` is what makes the horizon a hard
    GENERATION BUDGET; a ladder lets the session move to a wider trace instead.

    Doubling keeps the rung count logarithmic and bounds the wasted width at
    <2x, which matters more here than for dFlash: the MTP masks are rebuilt on
    the host and uploaded at ``[1, 1, P_v, S_k]`` on EVERY replay, so width is
    per-iteration bandwidth, not just device footprint.

    Pure: config-time arithmetic, no device and no model instance.
    """

    def _r(n):
        return ((int(n) + bucket - 1) // bucket) * bucket

    tail = int(draft_len) + 2
    smallest = _r(tail)
    largest = _r(int(max_context) + tail)
    ladder, rung = [], smallest
    while rung < largest:
        ladder.append(rung)
        rung = _r(rung * 2)
    ladder.append(largest)
    if max_rungs is not None and len(ladder) > int(max_rungs):
        # Keep the LARGEST rungs: dropping a small rung only widens short
        # prompts, while dropping the largest would leave long ones uncovered.
        ladder = ladder[-int(max_rungs) :]
    return ladder


def _dflash_drafter_config(snapshot):
    """Read the drafter checkpoint's HF config. Pure: file read, no device."""
    import json as _json

    with open(os.path.join(snapshot, "config.json")) as fh:
        cfg = _json.load(fh)
    return cfg.get("text_config") or cfg


def _dflash_mesh_tp():
    """Tensor-parallel width from ``MESH_DEVICE`` (e.g. ``P150x8`` -> 8).

    Returns 1 when unset or unparseable (Galaxy DP entries set no MESH_DEVICE).
    1 is the SAFE default here: ``local_kv = n_kv // tp``, so tp=1 yields the
    largest per-chip cache and therefore over-reserves rather than under-.
    """
    import re as _re

    m = _re.search(r"[xX](\d+)\s*$", os.environ.get("MESH_DEVICE", "") or "")
    return int(m.group(1)) if m else 1


def _hf_hub_cache_dirs():
    """Every hub root worth searching, most specific first.

    ``~/.cache/huggingface/hub`` is only the DEFAULT. A runner that points the
    cache elsewhere sets HF_HUB_CACHE (the tt-metal vllm-model-tests runner
    uses /mnt/MLPerf/huggingface/hub) or HF_HOME, and a drafter resolved by
    globbing the default alone is simply not found there -- which is rejected
    at config time, so the server fails to start rather than degrading.
    """
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(v)
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(os.path.join(home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen, out = set(), []
    for r in roots:
        r = os.path.expanduser(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _hf_snapshot_glob(repo_dir):
    """First snapshot of ``repo_dir`` across the hub roots, or None."""
    import glob as _glob

    for root in _hf_hub_cache_dirs():
        hits = _glob.glob(os.path.join(root, repo_dir, "snapshots", "*", ""))
        if hits:
            return hits[0]
    return None


def _hf_cache_is_writable():
    """Whether the HF hub cache can be written, i.e. whether a fetch can land.

    This, not HF_HUB_OFFLINE, is the condition that decides it. The CI weights
    share is mounted :ro by default and :rw when the operator asks for write
    access, and HF_HUB_OFFLINE is set because HF writes metadata (refs/main,
    .no_exist markers) on every HEAD call, which fails on a :ro mount with
    "Read-only file system". When the mount IS writable that objection is gone
    and a fetch is exactly what the operator asked for, so probe the mount
    rather than reading a flag that stands in for it.
    """
    for root in _hf_hub_cache_dirs():
        probe = root if os.path.isdir(root) else os.path.dirname(root.rstrip("/"))
        if probe and os.path.isdir(probe) and os.access(probe, os.W_OK):
            return True, root
    return False, None


def _hf_resolve_repo(repo_id, cache_dir_name):
    """Cache first, then the hub -- the way the TARGET model is resolved.

    transformers ``from_pretrained`` reads HF_HUB_CACHE and downloads what is
    missing. A drafter that only globbed the cache diverged from that: the
    target model appears and the drafter does not, and the failure reads as a
    missing checkpoint rather than "nobody fetched it".

    Returns a local path, or None when the repo is neither cached nor
    fetchable. A read-only cache is not an error here: the caller reports the
    miss with the env var that overrides it, which is more use than an HF
    stack trace.
    """
    hit = _hf_snapshot_glob(cache_dir_name)
    if hit:
        return hit
    writable, root = _hf_cache_is_writable()
    if not writable:
        logger.info(
            f"Gemma4: {repo_id} is not in the HF cache and the cache is read-only, "
            "so it will not be fetched; seed it or re-run with write access"
        )
        return None
    try:
        from huggingface_hub import constants as _hf_constants
        from huggingface_hub import snapshot_download

        # HF_HUB_OFFLINE is set for the read-only case, which does not apply to
        # a writable cache; disable it for this call only so the fetch is
        # allowed. Popping the ENV VAR is not enough and was the bug: hub reads
        # it once at import into constants.HF_HUB_OFFLINE, and every offline
        # gate goes through constants.is_offline_mode(), which returns that
        # module global. So a :rw run still raised LocalEntryNotFoundError
        # ("outgoing traffic has been disabled") and the drafter never
        # downloaded -- exactly the case the writable branch exists to serve.
        # Patch the global too, and restore both.
        saved = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        saved_offline = _hf_constants.HF_HUB_OFFLINE
        _hf_constants.HF_HUB_OFFLINE = False
        try:
            path = snapshot_download(repo_id=repo_id)
        finally:
            _hf_constants.HF_HUB_OFFLINE = saved_offline
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        logger.info(f"Gemma4: fetched {repo_id} into {root}")
        return path
    except Exception as exc:  # network, auth, or a repo that does not exist
        logger.warning(f"Gemma4: could not fetch {repo_id}: {type(exc).__name__}: {exc}")
        return None


def _target_weights_dir():
    """Local directory holding the TARGET model's safetensors.

    The drafter's embedding loader reads the target's embed_tokens straight out
    of the checkpoint, so it needs a real directory. ``MODEL_WEIGHTS_DIR`` is
    what tt-inference-server exports, but tt-metal's own vLLM CI does not set
    it: there ``HF_MODEL`` is a REPO ID and vLLM resolves the snapshot itself.
    Relying on the env var alone built the path "None/model.safetensors.index.json"
    and took the engine down during warmup after the drafter had downloaded
    fine. Fall back through the same places the rest of gemma4 looks, ending at
    the hub cache lookup the drafter itself uses.
    """
    cand = os.environ.get("MODEL_WEIGHTS_DIR")
    if cand and os.path.isdir(cand):
        return cand
    for env in ("HF_MODEL", "GEMMA4_MODEL_PATH"):
        cand = os.environ.get(env)
        if cand and os.path.isdir(cand):
            return cand
        # A repo id ("google/gemma-4-31B-it") resolves through the hub cache,
        # which is exactly how the leg already found the drafter.
        if cand and "/" in cand and not os.path.isabs(cand):
            hit = _hf_snapshot_glob("models--" + cand.strip("/").replace("/", "--"))
            if hit:
                return hit
    return None


def _dflash_default_snapshot():
    """Locate the z-lab drafter snapshot, cached or fetched (target parity)."""
    return _hf_resolve_repo("z-lab/gemma-4-31B-it-DFlash", "models--z-lab--gemma-4-31B-it-DFlash")


def _assistant_default_snapshot(hf_model):
    """Resolve the it-assistant (MTP drafter) checkpoint path.

    On the server ``HF_MODEL`` is a LOCAL weights-symlink dir, so the harness
    default ``f"{HF_MODEL}-assistant"`` yields a bogus path that
    AutoConfig.from_pretrained rejects (HFValidationError). Resolve robustly:
    (1) if ``{hf_model}-assistant`` is an existing dir, use it; else (2) infer
    the size (12B/31B) from the model string and glob the HF cache snapshot.
    """
    cand = f"{hf_model}-assistant"
    if os.path.isdir(cand):
        return cand
    size = "12B" if "12B" in str(hf_model) else "31B"
    hit = _hf_resolve_repo(f"google/gemma-4-{size}-it-assistant", f"models--google--gemma-4-{size}-it-assistant")
    return hit if hit else cand


def _spec_first_slot(empty_slots):
    """The state slot a B=1 speculative session belongs to, or None.

    The runner passes ``empty_slots`` to prefill and later calls
    ``release_request(slot)`` with the SAME slot, so this is what lets the
    model tell its own request's release from another's. None when the runner
    supplied nothing, which is the pre-existing single-session case.
    """
    if not empty_slots:
        return None
    try:
        return int(list(empty_slots)[0])
    except (TypeError, ValueError):
        return None


def _reserve_spec_ring_headroom(sliding_window, verify_width, where):
    """Reserve bounded-ring headroom for speculative candidate writes.

    A ring of exactly ``sliding_window`` is correct for plain decode, but a
    packed verify writes candidates at p+1..p+K BEFORE attention runs, and slot
    (p+j)%W holds position p+j-W, which is still inside the live window. Those
    writes evict history that an earlier candidate query in the SAME forward
    still needs, and masking future candidates cannot bring it back.

    The ring must stay a power of two (chunk starts must be multiples of both
    the ring and SDPA's q_chunk_size), so the smallest legal ring larger than
    the window is twice the window -- which also clears any K up to the window.
    Set through the env because the ring is read process-wide by
    ``bounded_ring_modulo`` from the model and trace paths, which do not know
    whether speculation is on. ``setdefault`` leaves an operator's own value
    alone.

    Only the demo used to set this, so a SERVER ran bounded sliding plus
    speculation on an exact-window ring and corrupted from the first token at
    K>1 (tt-metal#56048 review 3).
    """
    from models.demos.gemma4.tt.attention import _RING_HEADROOM_BLOCK, SPEC_RING_HEADROOM_ENV

    if sliding_window is None:
        return
    window = int(sliding_window)
    if window <= 0 or window % _RING_HEADROOM_BLOCK:
        return
    if os.environ.get(SPEC_RING_HEADROOM_ENV):
        return
    # NOT reserved automatically. The ring must stay a power of two, so the
    # smallest legal headroom DOUBLES it, and the bounded pool is sized
    # (ring/block)*max_batch for EVERY sliding layer -- 50 of them on 31B. That
    # doubling OOMs the shipped P150x8 config during KV allocation, so it
    # cannot be switched on by default; fitting it needs the full-attention
    # pool (GEMMA4_MAX_TOKENS_ALL_USERS) reduced to pay for it.
    #
    # Warn instead of proceeding silently: on an exact-window ring a packed
    # verify writes candidates at p+1..p+K into slots still holding live
    # window positions, so drafts corrupt from the first token at K>1
    # (tt-metal#56048 review 3). Loud, with the knob named, beats wrong tokens.
    blocks = window // _RING_HEADROOM_BLOCK
    logger.warning(
        f"{where}: bounded sliding with an EXACT-window ring ({window}) and "
        f"speculation (verify width {verify_width}). A packed verify writes "
        f"candidates at p+1..p+{verify_width} into slots that still hold live "
        f"window positions, which corrupts drafts at width > 1. Set "
        f"{SPEC_RING_HEADROOM_ENV}={blocks} to double the ring, and lower "
        "GEMMA4_MAX_TOKENS_ALL_USERS to pay for it -- the bounded pool is "
        "sized per sliding layer and doubling it OOMs the default config."
    )


class Gemma4DFlashForCausalLM(Gemma4ForCausalLM):
    """Gemma4 with the z-lab dFlash block-diffusion drafter, serving at B=1.

    Session pattern (DiffusionGemma precedent): prefill captures the residual
    taps (untraced -- traced prefill REPLAYS skip python-side hooks), the
    first decode call seeds the fused decoder (drafter ctx ingest + one-time
    trace capture at this request's horizon).

    PERF: each vLLM decode step runs a tight INTERNAL loop of dFlash iterations
    and commits a BLOCK of up to ``_SPEC_BLOCK`` tokens, so vLLM's per-STEP host
    overhead is amortized over the whole block (like DiffusionGemma's 256-token
    canvas) instead of being paid once per ~5-token iteration. This makes the server
    decode rate device-bound and ~metal-parity (measured 31B P150X8: code
    ~62 tok/s / prose ~47, vs metal demo 68/55 and baseline serving 26). The
    runner-supplied per-token inputs are advisory: the fused decoder owns the
    request's anchor/position state.
    """

    _SPEC_V = min(int(os.environ.get("GEMMA4_DFLASH_VERIFY", "7")), 15)
    _SPEC_N = _SPEC_V + 1
    # Server BLOCK size: one vLLM decode step runs a tight INTERNAL loop of
    # dFlash iterations and emits up to this many committed tokens, amortizing
    # vLLM's per-step host overhead over the whole block.
    #
    # MEASURED (supersedes an earlier "~135 ms/iter" claim, which was wrong --
    # it was a per-STEP number mislabelled per-iteration, taken before
    # async_scheduling). Fitting step_time = H + (B/acceptance)*D over three
    # block sizes on a P150x8 31B dFlash server (ISL 128, osl 1024, conc 1):
    #     B=64  88.0 tok/s/u  acc 5.61  11.40 iters/step  727 ms/step
    #     B=32  81.2 tok/s/u  acc 5.24   6.11 iters/step  394 ms/step
    #     B=8   57.0 tok/s/u  acc 5.04   1.59 iters/step  140 ms/step
    #   -> H = 39.4 ms per vLLM step, D = 59.9 ms per spec iteration
    #      (residuals -0.7% / +2.8% / -4.3%)
    # A whole iteration is only ~60 ms, so 135 ms/iter was impossible. At B=64
    # the host overhead is 5.5% of the step, and B=8 costs ~35% throughput --
    # the block loop still pays for itself, but far less than the old number
    # implied. Note H is WALL-CLOCK per step; viztracer puts plugin CPU work at
    # ~4 ms, so most of H is waiting (dispatch/IPC/queueing), not compute, and
    # is plausibly recoverable by a plugin-driven spec loop
    # (vllm-tt-plugin#110).
    _SPEC_BLOCK = int(os.environ.get("GEMMA4_DFLASH_SERVE_BLOCK", "64"))

    model_capabilities = {
        **Gemma4ForCausalLM.model_capabilities,
        # Async overlaps host scheduling with the device decode for the ADAPTIVE
        # BATCHED baseline fallback (conc>1), which returns raw device output the
        # runner reads on the deferred pipeline -- baseline-entry parity (sync
        # cost the fallback ~20-35% at conc-32). The plugin scheduler carries
        # each step's block decision on its SchedulerOutput, so the solo spec
        # block step (which returns committed host tokens with no read to
        # overlap) stays correct under the async schedule/commit lag. Kill
        # switch: GEMMA4_SUPPORTS_ASYNC_DECODE=0.
        "supports_async_decode": os.environ.get("GEMMA4_SUPPORTS_ASYNC_DECODE", "1").lower() in ("1", "true", "yes"),
        "supports_sample_on_device": True,  # decode returns TOKENS (host)
        # A solo-decode vLLM step commits exactly _SPEC_BLOCK valid tokens
        # (short blocks are EOS-filled at a genuine stop; upstream trims).
        # GEMMA4_DFLASH_SERVE_BLOCK=1 turns block-output OFF, so this impl can
        # also be deployed as a plain batched baseline (max_num_seqs>1) for the
        # concurrency>1 / throughput operating point -- see decode_forward.
        "output_tokens_per_step": _SPEC_BLOCK,
        # -- plugin speculative contract admission (vllm-tt-plugin#125) -----
        # Master gate. The plugin reads these four only when the launch carries
        # a speculative_config, and refuses the launch if this is absent.
        "supports_spec_decode": True,
        # What this drafter needs of the runner. dFlash drafts ON DEVICE from
        # the target's hidden state, so device_propose + hidden_feed. It owns no
        # paged drafter cache (its context cache is a fixed window), so
        # paged_drafter_cache is deliberately NOT declared -- the plugin refuses
        # that requirement outright since the TT backend cannot allocate it.
        "spec_requirements": ("device_propose", "hidden_feed"),
        # The hidden state never leaves the device: the fused verify hands it to
        # the drafter in-body. Required whenever hidden_feed is required.
        "spec_hidden_handoff": ("on_device",),
        # output_tokens_per_step above 1 selects the block-output rail. The
        # contract rail is a separate class, Gemma4DFlashContractForCausalLM,
        # which declares output_tokens_per_step 1 itself.
        # ADAPTIVE block-output: emit the spec block only when decoding ALONE
        # (batch==1); batch>1 decodes as plain baseline (exactly 1 token per
        # request, width-1 row -- the adaptive scheduler reserved exactly one
        # placeholder for such steps; NO sentinel padding).
        # This lets ONE server run max_num_seqs>1 -- dFlash at conc-1, baseline
        # batched at conc>1 (never worse) -- instead of the static max_num_seqs=1
        # block-output deployment. The scheduler reserves the K-token block only
        # on a solo decode step. Off when block-output itself is off (BLOCK<=1).
        "tt_adaptive_block_output": _SPEC_BLOCK > 1,
        # Spec-capture DRAM frontier (GEMMA4_DFLASH_MAX_SPEC_ISL, 0 = no limit):
        # a prompt longer than this serves as plain baseline for its whole
        # lifetime (prefill_forward never arms a session -- see the ceiling gate
        # there), so the adaptive scheduler must reserve width-1 for it even on
        # solo decode steps. Declaring the SAME value here keeps the scheduler's
        # reservation and the model's emission in lockstep with no sentinel.
        # Only meaningful while the ADAPTIVE block path is live. With
        # GEMMA4_DFLASH_SERVE_BLOCK=1 (block output OFF -- the documented
        # throughput operating point) tt_adaptive_block_output is False, and
        # declaring a prompt frontier without it is rejected at config time:
        # "tt_adaptive_block_max_prompt_tokens requires tt_adaptive_block_output".
        # That made the documented throughput config fail to boot.
        "tt_adaptive_block_max_prompt_tokens": (
            int(os.environ.get("GEMMA4_DFLASH_MAX_SPEC_ISL", "0")) if _SPEC_BLOCK > 1 else 0
        ),
        # Total KV positions ONE solo block step may touch, which is NOT the
        # emitted width: the loop commits up to _SPEC_BLOCK tokens and the final
        # iteration additionally writes _SPEC_N = V+1 physical verification rows
        # past them (carried accepted tokens are inside that same extent).
        #
        # The plugin's own fallback is twice the emitted width, which bounds this
        # only while the block is at least as wide as the verification -- true at
        # the shipped default (64 vs 8), false at e.g. SERVE_BLOCK=2 with
        # VERIFY=7, where the step emits 2 and writes 8 rows and the fallback
        # reserves 4. Declaring the real extent is what keeps a narrow block from
        # writing KV positions that have no request block
        # (vllm-tt-plugin#118 review, finding 2).
        "tt_block_kv_extent_tokens": (_SPEC_BLOCK + _SPEC_N) if _SPEC_BLOCK > 1 else 0,
    }

    # -- plugin speculative contract (vllm-tt-plugin#110 s.2) -----------------
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        """Declare what this model can speculate at ``(max_num_seqs, K)``.

        Pure and side-effect free: reads env + the drafter's ``config.json`` and
        allocates nothing. Deliberately does NOT read ``get_tt_*`` off
        ``vllm_config`` -- the platform stores those later and two of them
        default to 1 silently (#110 s.2).

        dFlash captures one fused trace per verify width of the width set; the
        supported K is the single configured verify width rather than a range,
        and this rail speculates for one request at a time.
        """
        del vllm_config  # nothing here is config-derived yet; see docstring

        # The PLUGIN owns these types (vllm_tt_plugin.spec_decode, merged in
        # vllm-tt-plugin#120) and its admission path does
        # ``isinstance(outcome, SpecReject)``. Returning a locally defined
        # look-alike would make that check False and a reject would be read as
        # a plan. Imported lazily: tt-metal must not import the plugin at
        # module scope.
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if max_num_seqs > 1:
            return SpecReject(
                reason=(
                    "Gemma4 dFlash speculation is single-stream: the fused verify trace is "
                    f"captured at B=1, so max_num_seqs={max_num_seqs} cannot speculate. Serve "
                    "max_num_seqs>1 through the adaptive block-output rail (batched steps run "
                    "plain baseline) or set max_num_seqs=1."
                ),
                supported_k=(),
            )

        verify = getattr(cls, "_SPEC_CONTRACT_K", None)
        if verify is None:
            verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))

        snapshot = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
        if not snapshot:
            # Reject at CONFIG time. Without this the drafter is resolved lazily
            # at the first request and a missing checkpoint kills the engine
            # there instead (EngineDeadError, no useful traceback).
            return SpecReject(
                reason=(
                    "Gemma4 dFlash drafter checkpoint not found: set GEMMA4_DFLASH_DRAFTER or "
                    "fetch it (hf download z-lab/gemma-4-31B-it-DFlash)."
                ),
                supported_k=(verify,),
            )
        try:
            cfg = _dflash_drafter_config(snapshot)
            n_layers = int(cfg["num_hidden_layers"])
            hidden = int(cfg["hidden_size"])
            head_dim = int(cfg["head_dim"])
            n_kv = int(cfg["num_key_value_heads"])
            block_size = int(os.environ.get("GEMMA4_DFLASH_BLOCK", cfg["block_size"]))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash drafter config unreadable at {snapshot}: {exc!r}. Point "
                    "GEMMA4_DFLASH_DRAFTER at a z-lab/gemma-4-31B-it-DFlash snapshot."
                ),
                supported_k=(verify,),
            )

        if not 1 <= verify < block_size:
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash verify count {verify} is outside [1, {block_size - 1}] "
                    f"for drafter block_size={block_size}. Set GEMMA4_DFLASH_VERIFY within that range."
                ),
                supported_k=(),
            )
        if requested_k < verify:
            # Captured shapes use the model's resolved count, independent of
            # how many drafts the scheduler requested.
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash verifies exactly {verify} drafts per iteration "
                    f"(GEMMA4_DFLASH_VERIFY); requested_k={requested_k} is below that and no "
                    "narrower verify bucket is captured."
                ),
                supported_k=(verify,),
            )

        tp = _dflash_mesh_tp()
        replicated = os.environ.get("GEMMA4_DFLASH_REPLICATED", "0") == "1"
        local_kv = n_kv if replicated else max(1, n_kv // tp)

        # Fixed per-session device bytes, PER CHIP (#110 expresses the Qwen3.6
        # figures per chip). Mirrors DFlashFusedDecoder.__init__ allocations:
        # every tensor below is bf16 (2 B) except the int64 position vectors.
        cap = int(os.environ.get("GEMMA4_DFLASH_CTX_CAP", "2048"))
        p_v = verify + 1  # packed-verify rows
        ctx_dev = cap * hidden * 2
        ctx_kv = 2 * n_layers * local_kv * cap * head_dim * 2
        ctx_pos = cap * 8
        fc_prev = p_v * hidden * 2
        commit_pos = p_v * 8
        extra_bytes_per_seq = ctx_dev + ctx_kv + ctx_pos + fc_prev + commit_pos

        return SpecPlan(
            effective_k=verify,
            lanes_per_request=1,
            extra_bytes_per_seq=extra_bytes_per_seq,
            # The drafter carries no paged KV that grows with the sequence: its
            # context cache is the fixed ``cap``-row window above.
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            drafter_target_cache_requires=(),
            # The baseline width-1 path exists (adaptive fallback), but it does
            # not carry the contract's per-row side tensors (accepted_counts /
            # num_valid_drafts). Gemma4DFlashContractForCausalLM.spec_plan sets
            # this to True.
            supports_narrow_decode=False,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_drafter = None
        self._spec_decoder = None
        self._spec_decoder_bucket = None  # packed-verify width bucket of the cached decoder (reuse key)
        self._spec_trace_ids = []
        self._spec_pending = None  # (taps, prompt_len) awaiting first decode
        # Identity of the request the pending/active session BELONGS to (see
        # _spec_pt_identity). The session is a single global slot, so a solo
        # decode must refuse taps captured for a different prompt.
        self._spec_pending_owner = None
        # Identity of the request an ACTIVE session belongs to. _spec_pending_owner
        # covers only the pre-bootstrap window; once bootstrapped it is cleared,
        # so without this an active session had NO ownership check at all and a
        # solo decode step scheduled for a non-owner re-pointed the fused decoder
        # at that request's page table and emitted a full block speculated from
        # the owner's residual taps -- wrong tokens against a single reserved
        # placeholder (review finding on tt-metal#56048 / vllm-tt-plugin#118.1).
        self._spec_active_owner = None
        self._spec_active = False
        # Tokens produced past a step's block width, delivered on the next step.
        self._spec_carry = []
        self._spec_horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048"))
        if self._bounded_sliding_kv_cache:
            _reserve_spec_ring_headroom(
                getattr(self._text_config(), "sliding_window", None), self._SPEC_N, "Gemma4DFlash"
            )
        # WIDTH SET (vllm-tt-plugin#110 s.8, tt-metal#56048 review step 3): the
        # verify widths are derived from max_model_len at CONFIG time and every
        # one is captured in warmup, so serving never captures and no captured
        # shape depends on a request. It also removes the reason _spec_budget_end
        # exists: the largest width covers max_model_len, so a request migrates
        # to a wider trace instead of being ended with an EOS at the horizon.
        # DEFAULT ON. Validated on a P150x8 31B server at 262144 max_model_len
        # with bounded sliding: mid-generation migration coherent, needle
        # retrieved at 121k and 238k, 32/32 needles at conc-32, and decode rate
        # within noise of the per-session path at both 4k and 121k (see the
        # commit message for the numbers). GEMMA4_DFLASH_WIDTH_SET=0 restores
        # per-session capture -- and with it the horizon generation cap.
        self._spec_width_set = os.environ.get("GEMMA4_DFLASH_WIDTH_SET", "1").lower() in ("1", "true", "yes")
        self._spec_width_ladder = None
        logger.info(
            f"Gemma4DFlash serving: V={self._SPEC_V} (N={self._SPEC_N}/step), "
            f"horizon={self._spec_horizon} new tokens/request, B=1 sessions"
        )

    # -- drafter ------------------------------------------------------------
    def _spec_get_drafter(self):
        if self._spec_drafter is not None:
            return self._spec_drafter
        from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter

        snap = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
        if not snap:
            raise RuntimeError("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")
        model0 = self.model[0]
        weights_dir = _target_weights_dir()
        if not weights_dir:
            raise RuntimeError(
                "dFlash drafter needs the TARGET model's weights directory to load "
                "its embedding table, and none could be resolved. Set "
                "MODEL_WEIGHTS_DIR (or point HF_MODEL/GEMMA4_MODEL_PATH at a local "
                "checkpoint, or seed the hub cache for it)."
            )

        def _embed_loader():
            import json as _json

            from safetensors import safe_open

            idx = _json.load(open(f"{weights_dir}/model.safetensors.index.json"))
            key = next(
                k
                for k in idx["weight_map"]
                if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
            )
            with safe_open(f"{weights_dir}/{idx['weight_map'][key]}", framework="pt") as f:
                return f.get_tensor(key)

        self._spec_drafter = DFlashDrafter(
            mesh_device=self.mesh_device,
            drafter_path=snap,
            target_embed_weight_loader=_embed_loader,
            mesh_config=model0.mesh_config,
            ccl_manager=model0.ccl_manager,
            tensor_cache_path=None,
        )
        return self._spec_drafter

    def warmup_model_decode(self, *args, **kwargs):
        """Warm the BATCHED BASELINE decode buckets -- in spec mode too.

        The adaptive model serves every concurrency>1 step through the plain
        batched baseline, so those buckets need warming exactly like a non-spec
        model. Only the SOLO spec step is a per-session fused trace, and that
        genuinely cannot be pre-warmed: its capture is per request and bucketed
        by prompt length (pv_bucket).

        This used to no-op in spec mode because the base warmup reader rejected
        the old sentinel-PADDED block output. That padding is gone -- the
        batched and no-session paths now return RAW DEVICE OUTPUT at width 1 --
        so the base warmup applies again. Skipping it made the first conc>1
        request of a run pay trace capture inside the measured window: on a
        P150x8 31B benchmark the 128/128 conc-32 cell showed TTFT 27.1 s cold
        against 11.7 s once the same shape had been driven first.

        Kill switch: GEMMA4_DFLASH_WARMUP_DECODE=0 restores the old no-op.
        """
        if self._SPEC_BLOCK <= 1:
            return super().warmup_model_decode(*args, **kwargs)
        if os.environ.get("GEMMA4_DFLASH_WARMUP_DECODE", "1").lower() not in (
            "1",
            "true",
            "yes",
        ):
            del args, kwargs
            self._decode_warmup_complete = True
            logger.info("Gemma4DFlash: batched decode warmup disabled by env")
            return None
        logger.info("Gemma4DFlash: warming batched baseline decode buckets")
        out = super().warmup_model_decode(*args, **kwargs)
        # Eager warmup prepares persistent dFlash allocations before either
        # ordinary prefill or ordinary decode records temporary addresses.
        if kwargs.get("enable_trace") is False:
            self._spec_get_drafter()
            if self._spec_width_set:
                self._spec_capture_width_set(kwargs.get("kv_cache"), kwargs.get("num_blocks"), prepare_only=True)
        elif self._spec_width_set and kwargs.get("enable_trace"):
            self._spec_capture_width_set(kwargs.get("kv_cache"), kwargs.get("num_blocks"))
        return out

    def _spec_capture_width_set(self, kv_cache, num_blocks, *, prepare_only=False):
        """Capture one fused verify trace per width, before the first request.

        The decoder built here is PERSISTENT for the life of the server: the
        traces belong to its buffers, so every request reseeds this one instance
        rather than constructing its own. That is the cross-request reuse path
        (GEMMA4_DFLASH_DECODER_REUSE), re-measured byte-identical to per-request
        capture on a 31B server -- same acceptance, same replies -- so making it
        the only path costs nothing and saves the per-request capture.

        The scratch page table is all zeros (block 0) at the FULL per-request
        width, which is what sizes v_pt at its maximum; every request refreshes
        the contents. Writing a handful of verify rows into block 0 at positions
        [0, P_v) during capture is harmless: any request that later owns that
        block overwrites those positions with its own prefill before it decodes.
        """
        import time as _time

        from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder

        if self._SPEC_BLOCK <= 1 or kv_cache is None:
            return
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0))
        ladder = dflash_width_set(
            max_seq_len,
            num_blocks,
            horizon=self._spec_horizon,
            verify=self._SPEC_V,
            max_rungs=int(os.environ.get("GEMMA4_DFLASH_WIDTH_RUNGS", "0")) or None,
        )
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        blocks = int(num_blocks) if num_blocks else max(1, max_seq_len // 64)
        scratch_pt = torch.zeros(1, blocks, dtype=torch.int32)
        dec = self._spec_decoder
        if dec is None:
            dec = DFlashFusedDecoder(
                self.model[0],
                self._spec_get_drafter(),
                kv_layers,
                scratch_pt,
                verify_count=getattr(self, "_SPEC_CONTRACT_K", None),
            )
            self._spec_decoder = dec
            self._spec_width_ladder = ladder
        elif self._spec_width_ladder != ladder:
            raise RuntimeError("Gemma4 dFlash verify widths changed after preparation")
        if prepare_only:
            dec.prepare_widths(ladder)
            return
        t0 = _time.time()
        cost = dec.capture_widths(ladder)
        logger.info(
            f"Gemma4DFlash: captured {len(cost)} verify widths in {_time.time()-t0:.1f}s "
            f"(max_model_len={max_seq_len}, widths={ladder}, per-width={ {k: round(v, 2) for k, v in cost.items()} })"
        )

    def _spec_pending_is_mine(self, page_table):
        """True when the pending session was captured for the request whose
        page table this is. Unknown identity on either side (no page table)
        falls back to True: that is the pre-existing single-session behaviour,
        and the scheduler's mirror still owns the width contract."""
        owner = getattr(self, "_spec_pending_owner", None)
        cur = self._spec_pt_identity(page_table)
        if owner is None or cur is None:
            return True
        return owner == cur

    def _spec_active_is_mine(self, page_table):
        """True when the LIVE session belongs to the request whose page table
        this is. Unknown identity on either side falls back to True, matching
        _spec_pending_is_mine: the scheduler's mirror still owns the width
        contract, and a missing page table is not evidence of a hand-off."""
        owner = getattr(self, "_spec_active_owner", None)
        cur = self._spec_pt_identity(page_table)
        if owner is None or cur is None:
            return True
        return owner == cur

    def _spec_drop_session(self, why):
        """Drop the single global spec session (pending taps AND any live one).

        Called on every path that serves a prefill as plain baseline. The taps
        are only valid for the prompt they were captured from, and the session
        slot is global: leaving it armed lets an unrelated later solo decode
        bootstrap another prompt's residuals, and leaving a live session armed
        lets it outlive the request whose width the scheduler reserved. The
        plugin's scheduler mirrors exactly these transitions, so the reserved
        width and the emitted width stay in lockstep.
        """
        if self._spec_pending is not None or self._spec_active:
            logger.info(f"Gemma4DFlash: dropping spec session ({why})")
        self._spec_pending = None
        self._spec_pending_owner = None
        if self._spec_active:
            self._spec_release_decoder()
        # DISARM the residual-tap hook. capture() arms it with buffers=tap_bufs,
        # which are sized for the fused DECODE body (P_v rows), and nothing
        # disarms it when the session ends -- so it stays armed after any spec
        # request. A later prefill served as plain baseline then runs the eager
        # chunked forward with that hook still live and ttnn.copy's a full
        # prefill chunk into a P_v-row buffer, killing the engine:
        #   TT_FATAL: Input tensor shape Shape([1, 1, 4096, 5376]) does not
        #   match output tensor shape Shape([1, 1, 6, 5376])
        # (copy_device_operation.cpp:112 -- 4096 = prefill chunk, 6 = P_v at
        # GEMMA4_DFLASH_VERIFY=5). Reproduced on a P150x8 256K benchmark sweep at
        # the first point above GEMMA4_DFLASH_MAX_SPEC_ISL: 131072 served fine,
        # 196608 took the engine down. Disarming here covers every drop path.
        try:
            self.model[0].dflash_capture_taps(None)
        except Exception:
            pass

    # -- prefill: capture taps (untraced) ------------------------------------
    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        # Baseline pickup: dFlash spec is B=1 block-output. In a non-block-output
        # / throughput deployment (GEMMA4_DFLASH_SERVE_BLOCK=1) or for any
        # batched (concurrency>1) prefill, serve via the plain baseline path and
        # skip the drafter tap capture entirely.
        if self._SPEC_BLOCK <= 1 or (tokens is not None and int(tokens.shape[0]) != 1):
            # model0's dFlash tap-capture state is SHARED; a prior solo session
            # can leave it armed (with decode-sized buffers). Disarm before a
            # plain baseline prefill so the tap hook does not fire on it (a stale
            # buffer copy would shape-mismatch against the prefill hidden).
            try:
                self.model[0].dflash_capture_taps(None)
            except Exception:
                pass
            self._spec_drop_session("baseline prefill")
            if self._SPEC_BLOCK > 1:
                # gemma4's prefill KV-history write (_left_pad_kv_to_hist) is not
                # trace-safe; the dFlash spec prefill runs untraced for the same
                # reason. Keep the adaptive batched baseline prefill untraced too.
                kwargs["enable_trace"] = False
            return super().prefill_forward(*args, **kwargs)
        drafter = self._spec_get_drafter()
        model0 = self.model[0]
        # Boot warmup prefills feed all-zero dummy tokens (and warmup_prefill=1);
        # capturing their taps would seed the drafter ctx with garbage. Only the
        # REAL prompt prefill sets the pending spec session.
        is_warmup = bool(kwargs.get("warmup_prefill")) or (tokens is not None and int(tokens.abs().sum()) == 0)
        # Force EAGER prefill: the residual taps are captured by a python hook in
        # the eager forward, which a traced replay skips. enable_trace=False gates
        # the prefill-bucket trace; GEMMA4_CHUNKED_PREFILL_TRACE=0 (model spec)
        # gates the per-chunk trace so multi-chunk prefills (ISL > one chunk)
        # still fire the hook -- without it the drafter gets empty taps at ISL
        # above the chunk size and the request fails.
        kwargs["enable_trace"] = False
        if is_warmup:
            return super().prefill_forward(*args, **kwargs)
        # SPEC-ISL CEILING (GEMMA4_DFLASH_MAX_SPEC_ISL, 0=off): above this prompt
        # length the fused-verify capture no longer fits in DRAM alongside the
        # batched-baseline persistent buffers (max_num_seqs>1) -- the capture
        # allocation OOMs (bank_manager.cpp:462) and kills the engine. Serve such
        # requests as plain baseline through the existing solo-no-session adaptive
        # path instead: skip the tap capture entirely so decode finds no pending
        # session. Frontier measured on P150x8 @ ctx=262144:
        # max_num_seqs=32 -> 131072 OK / 196608 OOM; 16 -> 196608 OK / 229376 OOM;
        # 1 -> 253952 OK. Keeps full batch fallback capacity while long-context
        # requests degrade gracefully to baseline speed rather than crashing.
        _max_spec_isl = int(os.environ.get("GEMMA4_DFLASH_MAX_SPEC_ISL", "0"))
        if _max_spec_isl > 0:
            _pl = kwargs.get("prompt_lens")
            _n0 = int(_pl[0]) if _pl is not None else int(tokens.shape[1])
            if _n0 > _max_spec_isl:
                logger.info(
                    f"Gemma4DFlash: prompt {_n0} > spec ceiling {_max_spec_isl}; "
                    "serving as plain baseline (no spec session)"
                )
                self._spec_drop_session("prompt over spec ceiling")
                return super().prefill_forward(*args, **kwargs)
        model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
        try:
            out = super().prefill_forward(*args, **kwargs)
        finally:
            taps = model0.pop_dflash_taps()
            model0.dflash_capture_taps(None)
        prompt_lens = kwargs.get("prompt_lens")
        n = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        self._spec_pending = (taps, n)
        self._spec_pending_owner = self._spec_pt_identity(kwargs.get("page_table"))
        # The runner releases by STATE SLOT, not by page table, so record the
        # slot too: release_request(row) has to tell "my request finished" from
        # "some other request finished" (see release_request).
        self._spec_owner_slot = _spec_first_slot(kwargs.get("empty_slots"))
        self._spec_active = False
        return out

    # -- decode: one fused spec iteration per call ----------------------------
    def _spec_bootstrap(self, anchor_id, start, page_table, kv_cache, page_tables_per_layer=None):
        import time as _time

        from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder

        taps, n = self._spec_pending
        self._spec_pending = None
        self._spec_pending_owner = None
        if start != n:
            logger.warning(f"Gemma4DFlash: first decode start_pos {start} != prompt_len {n}")
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        model0 = self.model[0]
        # BOUNDED sliding target (auto-enabled at >=131072): install the hybrid
        # per-layer page tables so the fused verify's v_ptl / _pv_setup read the
        # small RING pool for sliding layers and the flat global table for full-
        # attention layers -- the exact set the metal harness installs via
        # build_hybrid_page_tables (dflash_isl_sweep.py). Without it the verify
        # falls through to the flat table for EVERY layer, so sliding layers read
        # the wrong physical blocks, every draft is rejected, and acceptance
        # collapses (~1 tok/iter) at >=131072. Unbounded (<131072) leaves it unset
        # (flat table is already correct), preserving the current path.
        if self._bounded_sliding_kv_cache and page_table is not None:
            # No fallback: continuing with the flat table makes sliding layers
            # read the wrong physical blocks and silently collapses acceptance
            # (~1 tok/iter) -- fail the request loudly instead (review finding
            # on tt-metal#56048).
            _ptpl = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
            _ptpl = self._pad_sliding_page_tables_for_bounded(_ptpl, kv_cache, authoritative=True)
            if not _ptpl:
                raise RuntimeError(
                    "Gemma4DFlash: bounded per-layer page-table install produced "
                    "no tables; refusing to speculate against the flat table"
                )
            model0._active_page_tables_per_layer = _ptpl
        pt = page_table[:1] if page_table is not None else None
        horizon = self._spec_horizon
        t0 = _time.time()
        # REUSE the cached fused decoder when the new request shares its packed-
        # verify width bucket (pv_sk buckets by start/1024): the captured trace
        # reads persistent input buffers, so refreshing the KV page tables +
        # re-ingesting the prompt taps + re-uploading the anchor/pv inputs
        # re-points it at this request WITHOUT the ~2.6 s capture and WITHOUT
        # allocating/freeing a fresh decoder's buffers (which fragments DRAM).
        # A bucket change (different prompt length band) releases + re-captures.
        dec = self._spec_decoder
        # A carry never crosses sessions, and this must happen BEFORE any branch
        # below can return: the width-set reseed path returns early, so a reset
        # placed only on the capture path let request B open with tokens dFlash
        # produced for request A.
        self._spec_carry = []
        # Cross-request decoder reuse: DEFAULT OFF, conservatively -- the two
        # paths are now EQUIVALENT, so OFF is simply the unchanged behaviour and
        # not a considered preference.
        #
        # The earlier rationale here ("reseeded sessions draft worse": capture
        # 4.2-5.5 vs reseeded 1.9-3.7 tokens/iter, reuse ON ~2x slower at
        # ISL >= 1024) was an ARTIFACT of the page-table lifetime bug and is
        # retracted. That bug corrupted the no-reuse branch only -- release
        # deleted the per-layer tables the bootstrap had just installed -- so it
        # made freshly CAPTURED sessions look bad and, when the numbers happened
        # to fall the other way, reuse look bad. reseed() was never the problem.
        #
        # Re-measured on a P150x8 31B server AFTER that fix, 40,624-token prompt,
        # 4 identical requests, temperature 0 -- byte-identical replies both ways:
        #   acceptance   OFF 3.05/4.12/4.16/4.16   ON 3.05/4.12/4.16/4.16
        #   wall/request OFF 13.8-13.9 s           ON 12.7-12.8 s
        #   median TPOT  OFF 29.16 ms              ON 30.16 ms  (random ISL 32768)
        # ON saves the ~1.1 s capture per request and costs nothing measurable in
        # TPOT, so GEMMA4_DFLASH_DECODER_REUSE=1 is a safe win on workloads that
        # stay inside one packed-verify width bucket; it is not the default only
        # because bucket churn re-captures anyway and the DRAM-fragmentation
        # question for long-lived servers has not been measured.
        # With the WIDTH SET every width is already captured on the persistent
        # decoder, so a request never captures: it selects its width, refreshes
        # the page tables and reseeds. A width the set does not cover would be a
        # configuration error (the largest rung covers max_model_len), so fall
        # through to a capture rather than serve a wrong width.
        if self._spec_width_set and dec is not None and pt is not None:
            w = dec.width_for(int(start))
            if w is not None:
                dec.refresh_page_tables(pt)
                dec.select_width(int(start))
                dec.prefill_ingest(taps, n)
                dec.reseed(int(anchor_id), int(start))
                self._spec_decoder_bucket = w
                self._spec_active = True
                self._spec_active_owner = self._spec_pt_identity(page_table)
                self._spec_first_step = True
                self._spec_last_pt = None
                # The largest captured width bounds the generation, not the
                # horizon: vLLM's own max_model_len stop arrives first.
                self._spec_budget_end = max(self._spec_width_ladder or [w]) - self._SPEC_N - VERIFY_WIDTH_MARGIN
                logger.info(
                    f"Gemma4DFlash session: width-set reseed {_time.time()-t0:.2f}s "
                    f"(anchor={int(anchor_id)}, start={start}, width={w})"
                )
                return
            # Should be unreachable: the largest rung covers max_model_len.
            # Fall back to per-session capture, and turn the width set OFF for
            # the rest of the process rather than leaving a decoder whose traces
            # nothing will release (the capture below replaces it).
            logger.warning(
                f"Gemma4DFlash: no captured verify width covers start={int(start)}; "
                "capturing one for this session and disabling the width set"
            )
            self._spec_width_set = False
            self._spec_width_ladder = None
        _reuse_ok = os.environ.get("GEMMA4_DFLASH_DECODER_REUSE", "0").lower() in ("1", "true", "yes")
        reused = (
            _reuse_ok
            and dec is not None
            and pt is not None
            and dec.pv_bucket(int(start), horizon) == self._spec_decoder_bucket
        )
        if reused:
            dec.refresh_page_tables(pt)
            dec.prefill_ingest(taps, n)
            dec.reseed(int(anchor_id), int(start))
        else:
            # Keep the per-layer tables installed above: the new decoder reads
            # them in __init__ (see _spec_release_decoder's drop_page_tables).
            self._spec_release_decoder(drop_page_tables=False)
            dec = DFlashFusedDecoder(
                model0, self._spec_get_drafter(), kv_layers, pt, verify_count=getattr(self, "_SPEC_CONTRACT_K", None)
            )
            dec.prefill_ingest(taps, n)
            dec.capture(int(anchor_id), int(start), max_new=horizon)
            self._spec_decoder = dec
            self._spec_decoder_bucket = dec.pv_bucket(int(start), horizon)
        self._spec_active = True
        self._spec_active_owner = self._spec_pt_identity(page_table)
        self._spec_first_step = True
        self._spec_last_pt = None
        # verify masks/tables were sized for this horizon; past it the packed
        # verify would attend past its capture -- end the request cleanly then.
        self._spec_budget_end = int(start) + horizon - self._SPEC_N - 1
        logger.info(
            f"Gemma4DFlash session: {'REUSE' if reused else 'capture'} {_time.time()-t0:.2f}s "
            f"(anchor={int(anchor_id)}, start={start}, bucket={self._spec_decoder_bucket})"
        )

    def _spec_release_decoder(self, drop_page_tables=True, *, teardown=False):
        dec = self._spec_decoder
        if not teardown and self._spec_width_set and getattr(dec, "_pv_widths", None):
            # Session release preserves startup widths. Final teardown bypasses
            # retention and releases their traces while the mesh is open.
            self._spec_active = False
            self._spec_active_owner = None
            self._spec_decoder_bucket = None
            if drop_page_tables:
                try:
                    if hasattr(self.model[0], "_active_page_tables_per_layer"):
                        del self.model[0]._active_page_tables_per_layer
                except Exception:
                    pass
            return
        self._spec_decoder = None
        self._spec_decoder_bucket = None
        self._spec_active = False
        self._spec_active_owner = None
        # Drop the bounded per-layer tables installed for this session so a later
        # BATCHED baseline decode (adaptive fallback) rebuilds its own set instead
        # of reading this request's stale ring tables. Re-installed on next
        # bootstrap; kept alive DURING the session because refresh reads it.
        #
        # ``drop_page_tables=False`` is for the one caller that releases the OLD
        # decoder AFTER installing the NEW request's tables (_spec_bootstrap's
        # no-reuse branch). Dropping them there deletes the install that the
        # DFlashFusedDecoder constructed on the next line depends on: its v_ptl
        # comes from model._active_page_tables_per_layer, so the new session
        # would fall back to the FLAT table for sliding layers and decode
        # garbage at long context.
        if drop_page_tables:
            try:
                if hasattr(self.model[0], "_active_page_tables_per_layer"):
                    del self.model[0]._active_page_tables_per_layer
            except Exception:
                pass
        if dec is None:
            return
        tids = {id(t): t for t in [getattr(dec, "trace", None)] if t is not None}
        for _rec in (getattr(dec, "_pv_widths", None) or {}).values():
            if _rec.get("trace") is not None:
                tids.setdefault(id(_rec["trace"]), _rec["trace"])
        for _tid in tids.values():
            try:
                ttnn.release_trace(self.mesh_device, _tid)
            except Exception as e:
                logger.warning(f"Gemma4DFlash: trace release failed: {e!r}")
        for attr in ("ctx_k", "ctx_v"):
            for t in getattr(dec, attr, None) or []:
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        for _rec in (getattr(dec, "_pv_widths", None) or {}).values():
            for _t in (_rec.get("pv_iota"), *(_rec.get("cache_by_type") or {}).values()):
                if _t is not None:
                    try:
                        _t.deallocate(True)
                    except Exception:
                        pass
        for attr in (
            "ctx_dev",
            "fc_prev",
            "merge_idx",
            "commit_pos",
            "noise_rows",
            "anchor_row",
            "anchor_tok",
            "blk_pos",
            "mask_full",
            "mask_slide",
            "pv_iota",
            "pv_mask_slide",
            "pv_pos",
            "pv_widx_all",
            "out_ids",
        ):
            t = getattr(dec, attr, None)
            if t is not None:
                try:
                    t.deallocate(True)
                except Exception:
                    pass

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        start_pos = kwargs.get("start_pos")
        if start_pos is None and len(args) > 1:
            start_pos = args[1]
        if tokens is None:
            raise ValueError("Gemma4DFlash decode expects token input")
        # REAL request count, not the padded wire width (see _spec_real_batch).
        batch = self._spec_real_batch(tokens, start_pos)
        # Throughput mode (GEMMA4_DFLASH_SERVE_BLOCK=1 -> block-output OFF): plain
        # batched baseline at width 1, no spec, no padding.
        if self._SPEC_BLOCK <= 1:
            return super().decode_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        # Adaptive block-output: a BATCHED decode step (concurrency>1) runs plain
        # baseline and returns a host block padded to the reserved width K, so the
        # width-1 row (one valid token per request, no padding). The scheduler
        # reserved a single placeholder for this batched step (see TTScheduler),
        # matching the one real token per row. A solo request that just joined a
        # batch drops its dFlash session first -- baseline then owns its KV from
        # vLLM's committed position.
        if batch != 1:
            if self._spec_active or self._spec_pending is not None:
                self._spec_pending = None
                self._spec_release_decoder()
            # Disarm any shared tap capture so the baseline decode forward does
            # not fire the dFlash tap hook (see prefill_forward).
            try:
                self.model[0].dflash_capture_taps(None)
            except Exception:
                pass
            # Return the RAW device output (honoring read_from_device): the
            # runner's read_decode_output/process_decode_output_host/
            # _get_output_tokens pipeline converts, trims to the real batch, and
            # samples exactly as for a plain baseline model -- and under async
            # scheduling the deferred read overlaps the next step's host
            # scheduling (the whole point of the batched fallback). The width-1
            # rows commit through the adaptive scheduler's non-block path.
            return super().decode_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        anchor_from_runner = int(tokens.reshape(-1)[0])
        if self._spec_pending is not None and not self._spec_pending_is_mine(kwargs.get("page_table")):
            # The pending taps were captured for a DIFFERENT request (its
            # owner finished or was aborted before it ever decoded).
            # Bootstrapping them here would speculate from another prompt's
            # residuals and another prompt's length -- wrong tokens, not just
            # a wrong width. Drop them and serve this request as plain
            # baseline, which is also the width the scheduler reserved (its
            # session mirror sees this request as a non-owner too).
            logger.warning(
                "Gemma4DFlash: pending spec session belongs to another request; " "serving this one as plain baseline"
            )
            self._spec_pending = None
            self._spec_pending_owner = None
        if self._spec_pending is not None:
            start = int(start_pos.reshape(-1)[0]) if start_pos is not None else None
            self._spec_bootstrap(
                anchor_from_runner,
                start,
                kwargs.get("page_table"),
                kwargs.get("kv_cache"),
                page_tables_per_layer=page_tables_per_layer,
            )
        if self._spec_active and not self._spec_active_is_mine(kwargs.get("page_table")):
            # A solo decode step for a request that does NOT own the live
            # session. Reachable under async scheduling: the owner can reach
            # max_tokens and be skipped by upstream's num_output_placeholders
            # guard while its session is still armed, leaving another request
            # alone on the next step. Continuing here would re-point the fused
            # decoder at this request's page table and speculate from the
            # OWNER's residual taps -- wrong tokens, and a block width against
            # the single placeholder the scheduler reserved for a non-owner.
            logger.warning(
                "Gemma4DFlash: live spec session belongs to another request; "
                "releasing it and serving this one as plain baseline"
            )
            self._spec_release_decoder()
        if not self._spec_active:
            # Solo decode but no dFlash session -- e.g. a request that prefilled
            # BATCHED (concurrency>1, no tap capture) and is now decoding alone
            # after its peers finished. It cannot speculate (no taps), so serve
            # it as plain baseline: raw device output, one width-1 row through
            # the runner's baseline pipeline (same as the batched branch above).
            try:
                self.model[0].dflash_capture_taps(None)
            except Exception:
                pass
            return super().decode_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        dec = self._spec_decoder
        if not self._spec_first_step and anchor_from_runner != dec.anchor:
            logger.warning(
                f"Gemma4DFlash: runner anchor {anchor_from_runner} != session anchor "
                f"{dec.anchor}; trusting the session (advisory-input contract)"
            )
        # Refresh the verify page tables from vLLM's CURRENT per-request block
        # table when it CHANGES (the KV manager allocates a new block only every
        # ~block_size tokens).
        cur_pt = kwargs.get("page_table")
        if cur_pt is not None:
            row = cur_pt[:1] if cur_pt.dim() > 1 else cur_pt
            prev = getattr(self, "_spec_last_pt", None)
            if prev is None or not torch.equal(prev, row):
                dec.refresh_page_tables(row)
                self._spec_last_pt = row.clone()
        # BLOCK LOOP: run dFlash iterations back-to-back until this step's block
        # of up to _SPEC_BLOCK tokens is filled (or EOS / horizon). This is the
        # metal-demo tight loop, moved INSIDE one vLLM decode step so vLLM's
        # per-step host overhead is amortized over the whole block instead of
        # one ~5-token iteration -- the server rate then tracks the device-bound
        # speculation gain.
        eos = getattr(self.model[0].hf_config, "eos_token_id", 1)
        eos_set = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
        vocab = dec.drafter.vocab
        # CARRY: tokens a previous step produced past its block width. One
        # iteration commits up to V+1 tokens, so the loop below can pass
        # _SPEC_BLOCK mid-iteration. Dropping the excess (block[:K]) silently
        # DESYNCS the stream from the session: those tokens' KV is already
        # written and dec.start has advanced past them, so the model keeps
        # conditioning on tokens the caller never received -- a gap in the
        # delivered text, invisible in prose. Deliver them on the NEXT step
        # instead; they are already in KV, in order, so the sequence the model
        # conditions on and the sequence the caller receives stay identical.
        block = list(self._spec_carry)
        self._spec_carry = []
        while len(block) < self._SPEC_BLOCK:
            if self._spec_width_set:
                # Select the narrowest captured width that covers this position.
                # A request that has grown past its current width MOVES to the
                # next one; nothing but the active buffer set and trace changes,
                # because every width shares the drafter mirror, ctx cache and
                # commit state. This is what stops a captured width from acting
                # as a generation budget (review finding on tt-metal#56048).
                if dec.select_width(dec.start) is None:
                    logger.warning(
                        f"Gemma4DFlash: position {dec.start} past the widest captured "
                        "verify width; ending the request"
                    )
                    block.append(min(eos_set))
                    break
            elif dec.start >= self._spec_budget_end:
                block.append(min(eos_set))
                break
            accepted, bonus, produced = dec.step(first=self._spec_first_step)
            self._spec_first_step = False
            self._spec_iters = getattr(self, "_spec_iters", 0) + 1
            self._spec_tokens = getattr(self, "_spec_tokens", 0) + produced
            committed = list(accepted) + [bonus]
            oov = [t for t in committed if not 0 <= t < vocab]
            if oov:
                # The verify posterior produced an id outside the vocab -- a
                # sign of upstream corruption, not a normal decode event.
                # Substitute to keep the wire valid but SAY so (review finding
                # on tt-metal#56048: never mask this silently).
                if not getattr(self, "_spec_oov_warned", False):
                    self._spec_oov_warned = True
                    logger.warning(
                        f"Gemma4DFlash: {len(oov)} out-of-vocab committed id(s) "
                        f"(first={oov[0]}, vocab={vocab}); substituting. "
                        "Investigate verify integrity if this repeats."
                    )
                committed = [t if 0 <= t < vocab else int(bonus if 0 <= bonus < vocab else 1) for t in committed]
            block.extend(committed)
            if eos_set & set(committed):
                break
        if len(block) > self._SPEC_BLOCK:
            # Past the width: hold the tail for the next step rather than
            # dropping it (see CARRY above). An EOS anywhere in this block ends
            # the request, so a carry behind it is moot -- the scheduler trims
            # at the first stop token and release_request clears it.
            self._spec_carry = block[self._SPEC_BLOCK :]
            block = block[: self._SPEC_BLOCK]
        # Exactly-K valid tokens: a short block now happens only at a genuine
        # stop -- EOS committed, or a position past the widest captured verify
        # width -- so fill the tail with EOS; the scheduler trims committed
        # tokens at the first stop token, and the plugin no longer accepts
        # sentinel padding. (With the width set the old "horizon exhausted"
        # case is gone: the session migrates instead of ending.)
        out = torch.full((1, self._SPEC_BLOCK), min(eos_set), dtype=torch.int32)
        out[0, : len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    def read_decode_output(self, tt_out, async_read=False, *_, **__):
        # A SOLO SPEC BLOCK step returns committed host tokens from
        # decode_forward -- nothing to read, pass them straight through (no
        # events). Every other output is a DEVICE tensor: throughput mode, and
        # the adaptive batched / no-session baseline steps that now return raw
        # device output for the async read overlap. Route those to the base
        # reader (events under async_read) so the deferred pipeline reads them.
        import torch

        if self._SPEC_BLOCK > 1 and isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *_, **__)

    # -- plugin lifecycle hooks (block-output contract) -----------------------
    def note_state_slots_moved(self, moves) -> None:
        """The runner gathered per-slot state: ``moves`` is slot ``old -> new``.

        The B=1 session's owner slot is the identity ``release_request`` compares
        against, and the runner permutes slots between steps, so it has to follow
        the move or a release stops matching its own request. The runner passes
        the whole permutation at once because applying pairs one at a time can
        move the same owner twice (vllm-tt-plugin#118 review, finding 1).
        """
        owner = getattr(self, "_spec_owner_slot", None)
        if owner is None or not moves:
            return
        try:
            moved = moves.get(int(owner))
        except (AttributeError, TypeError, ValueError):
            return
        if moved is not None:
            self._spec_owner_slot = int(moved)

    def release_request(self, row: int) -> None:
        """Request finished. KEEP the cached fused decoder
        alive so the next request in the same packed-verify width bucket reuses
        it -- no ~2.6 s re-capture, no per-request buffer churn. The decoder is
        released on a bucket change (_spec_bootstrap) or at capture teardown
        (release_persistent_capture)."""
        owner_slot = getattr(self, "_spec_owner_slot", None)
        if owner_slot is not None and row is not None and int(row) != int(owner_slot):
            # ANOTHER request's slot. Adaptive serving admits several live
            # requests while this adapter keeps one session, so a release must
            # not tear down a session a DIFFERENT live request owns: that
            # request would then decode one baseline token against the full
            # block the scheduler reserved for it, and the scheduler rejects
            # the width.
            return
        it = getattr(self, "_spec_iters", 0)
        if it:
            tk = getattr(self, "_spec_tokens", 0)
            logger.info(f"Gemma4DFlash decode summary: {it} iters, {tk} tokens, " f"{tk/max(1,it):.2f} tokens/iter")
        self._spec_iters = 0
        self._spec_tokens = 0
        self._spec_pending = None
        self._spec_pending_owner = None
        self._spec_active = False  # session inactive, decoder retained for reuse
        # The retained decoder is reused across requests, so anything request
        # SCOPED must die here: a carry left behind is emitted as the next
        # request's first tokens.
        self._spec_carry = []
        self._spec_owner_slot = None

    def release_persistent_capture(self) -> None:
        self._spec_pending = None
        self._spec_pending_owner = None
        self._spec_carry = []
        try:
            self._spec_release_decoder(teardown=True)
        finally:
            super().release_persistent_capture()


def _dflash_env_flag(name, default):
    """True when ``name`` holds one of the on-values the GEMMA4_* gates accept."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


@dataclasses.dataclass
class RetainedProposal:
    """What one fused replay produced, kept for the step that verifies it.

    The fused body drafts K tokens after ``anchor_token`` and, in the same
    replay, runs the target over the anchor and those drafts. ``posterior``
    holds the target's argmax at each of the 1+K positions. The step that
    carries the drafts answers from it instead of running the device;
    ``consumed`` records that it answered once.
    """

    anchor_token: int
    anchor_position: int
    drafts: list
    posterior: list
    consumed: bool = False


class Gemma4DFlashContractForCausalLM(Gemma4DFlashForCausalLM):
    """dFlash speculation driven by the plugin's speculative-decoding contract.

    The block rail (:class:`Gemma4DFlashForCausalLM`) drafts, verifies and
    walks acceptance inside one engine step and emits a block of committed
    tokens. This class hands the same fused decoder to the runner one
    iteration at a time: ``output_tokens_per_step`` is 1, the runner builds
    the candidate block from the drafts this class proposed and owns the
    accept walk.

    Steps for one request that decodes alone:

    1. ``prefill_forward`` runs the prompt eagerly and keeps the residual taps
       of the drafter's target layers as ``_spec_pending``. A prompt split
       across scheduler steps accumulates its taps over those calls.
    2. The first ``decode_forward`` arrives as an ordinary step (narrow
       decode). It seeds the fused decoder from the taps (``_spec_bootstrap``),
       runs one fused replay and returns that replay's first posterior id as
       the sampled token. Nothing else touches the device.
    3. ``propose_draft_tokens`` commits the runner's accepted count and anchor,
       refreshes the verify page tables, runs one fused replay and keeps its
       drafts and posterior as a :class:`RetainedProposal`.
    4. The next ``decode_forward`` carries those drafts (``spec_mode`` set).
       It checks the block against the retained proposal and answers from the
       posterior without touching the device.
    5. Steps 3 and 4 repeat: one fused replay per step, never a second
       prefill.

    Speculation ends, and does not resume, when another request shares a
    step with the owner, when the owner was prefilled together with another
    request, when the runner samples on host, or when a proposal declines
    (see ``propose_draft_tokens``). The row then continues as plain decode,
    which is what the block rail does at batch size above one. A step the
    owner is not part of leaves its session alone: the scheduler keeps the
    drafts of a request it did not schedule and sends them on that request's
    next step, which the retained proposal must still answer.

    Identity: the request that owns the pending taps or the live session is
    known by its state slot (``empty_slots`` at prefill, moved by
    ``note_state_slots_moved``, closed by ``release_request``) together with
    a per-slot generation count, so a reused slot never matches a stale
    owner. The fused decoder addresses batch row 0 (``page_table[:1]`` in
    ``_spec_bootstrap``, ``lpt[0]`` in ``refresh_page_tables``), so only a
    request decoding at row 0 is bootstrapped.

    Block tables: vLLM's ``BlockTable.add_row`` and ``move_row`` write only a
    row's current block count, so the plugin's fixed-width slice can carry a
    previous occupant's block ids in the columns past a prompt's last block.
    Every prefill on this rail is eager and the eager paged fill stops at the
    valid length, so those columns are never addressed here;
    ``_dflash_mask_prefill_tables`` zeroes them anyway so a traced chunk
    cannot write another request's blocks.
    """

    _SPEC_CONTRACT_K = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))
    # The inherited width ladder, ring-headroom warning and generation budget
    # read these; on this rail the verify count is the contract K.
    _SPEC_V = _SPEC_CONTRACT_K
    _SPEC_N = _SPEC_CONTRACT_K + 1
    # The inherited prefill and warmup paths read ``_SPEC_BLOCK > 1`` as
    # "speculation is on". The value never reaches the wire here.
    _SPEC_BLOCK = max(2, int(os.environ.get("GEMMA4_DFLASH_SERVE_BLOCK", "64")))
    _DFLASH_ASYNC = _dflash_env_flag("GEMMA4_CONTRACT_ASYNC", "0")

    model_capabilities = {
        **Gemma4ForCausalLM.model_capabilities,
        "output_tokens_per_step": 1,
        # A step this class does not speculate on runs the plain decode; the
        # device sampler keeps that step from pulling [B, vocab] logits to host.
        "supports_sample_on_device": True,
        # Off by default: upstream vLLM refuses async scheduling for
        # method=custom_class, and the runner's overlap would let a proposal
        # run after the next verify was submitted.
        "supports_async_decode": _DFLASH_ASYNC,
        "supports_async_spec_decode": _DFLASH_ASYNC,
        "supports_spec_decode": True,
        # Drafts come from the device and the target hidden state never leaves
        # it, so the runner holds no handle and ``hidden`` is unused.
        "spec_requirements": ("device_propose",),
        "spec_hidden_handoff": ("on_device",),
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Admit any ``max_num_seqs``: speculation is decided per step.

        The inherited plan refuses ``max_num_seqs > 1`` because the fused
        verify packs one request's candidates into one batch row. That is a
        statement about a step, not about the server: a step with several live
        rows proposes nothing and runs the plain decode. The plan's arithmetic
        (effective_k, lanes, byte accounting) is inherited so it exists once.
        """
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        verify = cls._SPEC_CONTRACT_K
        if os.environ.get("GEMMA4_DFLASH_PACKED", "1") != "1":
            return SpecReject(
                reason=(
                    "Gemma4 dFlash on the speculative contract needs packed verification "
                    "(GEMMA4_DFLASH_PACKED=1): only the packed body writes effective_k + 1 KV "
                    "rows per step, which is the lookahead the runner reserves."
                ),
                supported_k=(),
            )
        for env in ("GEMMA4_DFLASH_WIDTH_SET", "GEMMA4_DFLASH_WARMUP_DECODE"):
            if not _dflash_env_flag(env, "1"):
                return SpecReject(
                    reason=(
                        f"Gemma4 dFlash on the speculative contract needs {env}=1: the verify "
                        "widths are prepared and captured in warmup, before ordinary trace "
                        "capture, and never during serving."
                    ),
                    supported_k=(verify,),
                )
        outcome = super().spec_plan(vllm_config, 1, requested_k)
        if not isinstance(outcome, SpecPlan):
            return outcome
        # A step with no draft anywhere runs at the plain decode's [B, 1] shape.
        return dataclasses.replace(outcome, supports_narrow_decode=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dflash_init_state()
        self._dflash_ring_advice()
        logger.info(
            f"Gemma4DFlash speculative contract: K={self._SPEC_CONTRACT_K}, one fused replay per "
            "solo step, output_tokens_per_step=1"
        )

    def _dflash_init_state(self):
        self._dflash_retained = None
        self._dflash_slot_gen = {}
        self._dflash_pending_owner = None
        self._dflash_live_owner = None
        self._dflash_owner_tables = None
        self._dflash_last_pt = None
        self._dflash_stale_rows = set()
        self._dflash_steps = 0
        self._dflash_committed = 0
        self._dflash_ring = self._dflash_ring_geometry()

    def _dflash_ring_geometry(self):
        """``(ring, window)`` of the bounded sliding layers, or None when none is bounded."""
        models = getattr(self, "model", None) or []
        layers = getattr(models[0], "layers", None) if models else None
        for layer in layers or []:
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            ring = getattr(cfg, "cache_position_modulo", None)
            if ring is not None:
                return int(ring), int(cfg.sliding_window)
        return None

    def _dflash_ring_advice(self):
        if self._dflash_ring is None:
            return
        ring, window = self._dflash_ring
        p_v = self._SPEC_CONTRACT_K + 1
        if ring - window >= p_v - 1:
            return
        from models.demos.gemma4.tt.attention import _RING_HEADROOM_BLOCK, SPEC_RING_HEADROOM_ENV

        logger.warning(
            f"Gemma4DFlash speculative contract: the bounded ring ({ring}) equals the sliding "
            f"window ({window}) and a verify writes P_v={p_v} rows, so proposals decline once a "
            f"request passes position {ring - p_v} and it continues as plain decode. "
            f"{SPEC_RING_HEADROOM_ENV}={window // _RING_HEADROOM_BLOCK} (ring {2 * window}) would "
            "lift that, once the bounded page tables are sized from the ring (tt-metal#57573); "
            "it also needs a lower GEMMA4_MAX_TOKENS_ALL_USERS to fit."
        )

    # -- identity: state slot plus generation ---------------------------------
    def _dflash_open_owner(self, slot):
        if slot is None:
            return None
        slot = int(slot)
        gen = self._dflash_slot_gen.get(slot, 0) + 1
        self._dflash_slot_gen[slot] = gen
        return (slot, gen)

    def _dflash_owns(self, owner, row):
        if owner is None or row is None:
            return False
        slot, gen = owner
        return int(row) == slot and self._dflash_slot_gen.get(slot) == gen

    def note_state_slots_moved(self, moves) -> None:
        if not moves:
            return
        try:
            moved = {int(old): int(new) for old, new in dict(moves).items()}
        except (AttributeError, TypeError, ValueError):
            return
        gens = dict(self._dflash_slot_gen)
        for old, new in moved.items():
            gens[new] = self._dflash_slot_gen.get(old, 0)
        self._dflash_slot_gen = gens

        def follow(owner):
            if owner is None or owner[0] not in moved:
                return owner
            return (moved[owner[0]], owner[1])

        self._dflash_pending_owner = follow(self._dflash_pending_owner)
        self._dflash_live_owner = follow(self._dflash_live_owner)
        self._spec_owner_slot = self._dflash_live_owner[0] if self._dflash_live_owner else None

    def release_request(self, row: int) -> None:
        if row is None:
            self._dflash_drop_pending()
            self._dflash_end_session()
            return
        row = int(row)
        pending_mine = self._dflash_owns(self._dflash_pending_owner, row)
        live_mine = self._dflash_owns(self._dflash_live_owner, row)
        # Close the slot's generation: a request prefilled into it opens a new one.
        self._dflash_slot_gen[row] = self._dflash_slot_gen.get(row, 0) + 1
        if pending_mine:
            self._dflash_drop_pending()
        if live_mine:
            self._dflash_end_session()

    def release_persistent_capture(self) -> None:
        self._dflash_retained = None
        self._dflash_live_owner = None
        self._dflash_pending_owner = None
        self._dflash_owner_tables = None
        super().release_persistent_capture()

    # -- session state ---------------------------------------------------------
    def _dflash_drop_pending(self):
        pending = self._spec_pending
        self._spec_pending = None
        self._spec_pending_owner = None
        self._dflash_pending_owner = None
        if pending is None:
            return
        for tap in pending[0]:
            try:
                tap.deallocate(True)
            except Exception:
                pass

    def _dflash_end_session(self):
        if self._dflash_steps:
            # One line per session, comparable with the block rail's decode
            # summary: every step of the session counts, the first ordinary
            # step included, and tokens are what the runner committed.
            logger.info(
                f"Gemma4DFlash contract summary: {self._dflash_steps} steps, {self._dflash_committed} tokens, "
                f"{self._dflash_committed / self._dflash_steps:.2f} tokens/step"
            )
        self._dflash_steps = 0
        self._dflash_committed = 0
        self._dflash_retained = None
        self._dflash_live_owner = None
        self._dflash_owner_tables = None
        self._dflash_last_pt = None
        self._spec_owner_slot = None
        self._spec_carry = []
        if self._spec_active:
            self._spec_release_decoder()
        self._spec_active = False

    def _dflash_owner_in(self, slots):
        return any(self._dflash_owns(self._dflash_live_owner, slot) for slot in slots)

    def _dflash_leave_speculation(self, why, slots=None):
        """Drop the pending taps and end the live session.

        ``slots`` are the state slots this step serves. When given and none of
        them owns the live session, the session is kept for the owner's next
        step (see the class docstring).
        """
        keep = self._spec_active and slots is not None and not self._dflash_owner_in(slots)
        if self._spec_pending is not None or (self._spec_active and not keep):
            logger.info(f"Gemma4DFlash speculative contract: leaving speculation ({why})")
        self._dflash_drop_pending()
        if not keep:
            self._dflash_end_session()
        self._dflash_disarm_taps()

    def _dflash_disarm_taps(self):
        try:
            self.model[0].dflash_capture_taps(None)
        except Exception:
            pass

    def _dflash_mark_stale(self, row):
        """The plain decode's device-resident token and position for ``row`` are stale.

        The traced decode keeps each row's token and position on the device
        and reloads them from the host only on a batch layout change. A row
        answered from the drafter never went through that trace, so its next
        plain decode must reload every row's inputs from the host
        (``_dflash_plain_kwargs``); under async decode the merge keeps host
        inputs for the rows listed in ``_slots_prefilled_since_decode``.
        """
        self._dflash_stale_rows.add(int(row))
        slots = getattr(self, "_slots_prefilled_since_decode", None)
        if slots is None:
            slots = self._slots_prefilled_since_decode = set()
        slots.add(int(row))

    def _dflash_plain_kwargs(self, kwargs):
        """Keyword arguments for a plain decode, with a host reload when a drafter-served row needs one."""
        if not self._dflash_stale_rows:
            return kwargs
        self._dflash_stale_rows.clear()
        kwargs = dict(kwargs)
        kwargs["reset_batch"] = True
        return kwargs

    def _dflash_note_step(self, row, kwargs, page_tables_per_layer):
        """Record the owner's tables from this step for the next page-table refresh."""
        self._dflash_owner_tables = (kwargs.get("page_table"), page_tables_per_layer, kwargs.get("kv_cache"))
        self._dflash_mark_stale(row)

    # -- prefill: capture taps, accumulate over scheduler chunks --------------
    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        if args:
            kwargs["tokens"] = args[0]
            args = args[1:]
        tokens = kwargs.get("tokens")
        rows = int(tokens.shape[0]) if tokens is not None else 1
        prompt_lens = kwargs.get("prompt_lens")
        end = int(prompt_lens[0]) if prompt_lens is not None else (int(tokens.shape[1]) if tokens is not None else 0)
        start_pos = kwargs.get("start_pos")
        cached = int(list(start_pos)[0]) if start_pos is not None else 0
        kwargs["page_table"], page_tables_per_layer = self._dflash_mask_prefill_tables(
            kwargs.get("page_table"), page_tables_per_layer, prompt_lens, tokens, kwargs.get("kv_cache")
        )
        # Every prefill on this rail is eager: the residual taps come from a
        # python hook that a traced replay skips, and a plain prefill must not
        # run traced next to an eager one against the same persistent buffers.
        kwargs["enable_trace"] = False
        is_warmup = bool(kwargs.get("warmup_prefill")) or (tokens is not None and int(tokens.abs().sum()) == 0)
        if is_warmup:
            self._dflash_disarm_taps()
            return Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        slot = _spec_first_slot(kwargs.get("empty_slots"))
        continues = (
            cached > 0
            and self._spec_pending is not None
            and self._dflash_owns(self._dflash_pending_owner, slot)
            and int(self._spec_pending[1]) == cached
        )
        if not continues:
            self._dflash_drop_pending()
        speculable = rows == 1 and self._dflash_prompt_speculable(end)
        if speculable and cached > 0 and not continues:
            # Another prefill replaced this request's earlier taps. This chunk's
            # taps alone would seed the drafter as if they were the whole prompt.
            speculable = False
        if speculable and cached > 0 and align_num_cached_tokens_to_sdpa([cached])[0] != cached:
            # The generator re-prefills the tokens between the aligned start and
            # the chunk start, so their taps would appear twice and shift every
            # later row of the drafter's window.
            speculable = False
        if not speculable:
            self._dflash_drop_pending()
            self._dflash_disarm_taps()
            return Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        if not continues:
            self._dflash_pending_owner = self._dflash_open_owner(slot)
        drafter = self._spec_get_drafter()
        model0 = self.model[0]
        model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
        try:
            out = Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        finally:
            taps = model0.pop_dflash_taps()
            model0.dflash_capture_taps(None)
        previous = list(self._spec_pending[0]) if continues and self._spec_pending is not None else []
        taps = self._dflash_trim_taps(previous + list(taps), len(drafter.target_layer_ids))
        self._spec_pending = (taps, end)
        self._spec_pending_owner = None
        self._spec_active = self._spec_active and self._dflash_live_owner is not None
        return out

    def _dflash_prompt_speculable(self, prompt_len):
        limit = int(os.environ.get("GEMMA4_DFLASH_MAX_SPEC_ISL", "0"))
        if limit > 0 and int(prompt_len) > limit:
            logger.info(
                f"Gemma4DFlash speculative contract: prompt {prompt_len} > GEMMA4_DFLASH_MAX_SPEC_ISL "
                f"{limit}; serving as plain decode"
            )
            return False
        return True

    def _dflash_trim_taps(self, taps, per_group):
        """Keep the newest tap groups that cover the drafter's context window.

        One group per prefill forward, ``per_group`` tensors each. Groups
        wholly below the last ``ctx_cap`` positions never reach the mirror
        (``prefill_ingest`` slices the window from the end), so they are freed
        here instead of accumulating across scheduler chunks.
        """
        if per_group <= 0 or len(taps) % per_group:
            return list(taps)
        cap = int(getattr(self._spec_decoder, "cap", 0) or os.environ.get("GEMMA4_DFLASH_CTX_CAP", "2048"))
        groups = [taps[i : i + per_group] for i in range(0, len(taps), per_group)]
        kept, rows = [], 0
        for group in reversed(groups):
            if rows >= cap:
                for tap in group:
                    try:
                        tap.deallocate(True)
                    except Exception:
                        pass
                continue
            kept.append(group)
            rows += int(group[0].shape[2])
        return [tap for group in reversed(kept) for tap in group]

    def _dflash_mask_prefill_tables(self, page_table, page_tables_per_layer, prompt_lens, tokens, kv_cache):
        """Zero block-table columns past each prompt's last block.

        vLLM writes only a row's current block count, so the plugin's
        fixed-width slice can carry a previous occupant's block ids past that
        count. A fill that ran to the padded bucket would write through them
        into blocks another request may own. Bounded sliding layers address
        ring slots and are left alone.
        """
        if kv_cache is None or tokens is None:
            return page_table, page_tables_per_layer
        try:
            block_size = int(self._effective_paged_block_size(kv_cache) or 0)
        except Exception:
            block_size = 0
        if block_size <= 0:
            return page_table, page_tables_per_layer
        rows = int(tokens.shape[0])
        if prompt_lens is not None:
            ends = [int(n) for n in list(prompt_lens)][:rows]
        else:
            ends = [int(tokens.shape[1])] * rows
        first_free = [(end + block_size - 1) // block_size for end in ends]

        def mask(table):
            if not isinstance(table, torch.Tensor) or table.dim() != 2:
                return table
            table = table.clone()
            for row, col in enumerate(first_free):
                if row < int(table.shape[0]) and col < int(table.shape[1]):
                    table[row, col:] = 0
            if int(table.shape[0]) > len(first_free):
                table[len(first_free) :, :] = 0
            return table

        page_table = mask(page_table)
        if page_tables_per_layer:
            ring = set(self._sliding_layer_indices()) if self._bounded_sliding_kv_cache else set()
            page_tables_per_layer = [
                table if index in ring else mask(table) for index, table in enumerate(page_tables_per_layer)
            ]
        return page_table, page_tables_per_layer

    # -- decode ----------------------------------------------------------------
    @staticmethod
    def _dflash_inputs(args, kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        return tokens, start_pos

    @staticmethod
    def _dflash_rows(tokens):
        if tokens is None:
            return 1
        return int(tokens.shape[0]) if tokens.dim() > 1 else int(tokens.numel())

    @staticmethod
    def _dflash_live_rows(positions, rows):
        """Indices of rows that belong to real requests: the runner pads with -1."""
        if positions is None:
            return list(range(int(rows)))
        try:
            first = positions.reshape(int(rows), -1)[:, 0]
        except Exception:
            return list(range(int(rows)))
        return [int(r) for r in torch.nonzero(first >= 0).reshape(-1).tolist()]

    @staticmethod
    def _dflash_row(values, rows, row):
        """Row ``row`` of a ``[rows, W]`` or flat ``[rows]`` tensor as ints."""
        v = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        return [int(x) for x in v.reshape(int(rows), -1)[int(row)].tolist()]

    @staticmethod
    def _dflash_slot_of(row, kwargs):
        """The state slot row ``row`` reads on this step.

        The runner permutes per-slot state with ``slot_remap`` (``slot_remap[i]``
        is the slot whose state row ``i`` reads) and reports the move through
        ``note_state_slots_moved`` only after the model accepted the step, so
        during the step a request's row and its owner slot can differ.
        """
        remap = kwargs.get("slot_remap")
        if remap is None:
            return int(row)
        try:
            return int(remap[int(row)])
        except (IndexError, TypeError, ValueError):
            return int(row)

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        kwargs.pop("accepted_counts", None)  # consumed by propose, where the commit is
        kwargs.pop("draft_token_ids", None)
        tokens, start_pos = self._dflash_inputs(args, kwargs)
        rows = self._dflash_rows(tokens)
        live = self._dflash_live_rows(start_pos, rows)
        if spec_mode is None:
            return self._dflash_ordinary_step(args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live)
        return self._dflash_verify_step(args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live, num_valid)

    def _dflash_ordinary_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live):
        """An ordinary decode step: the runner asked for no verification.

        With one live row this is where speculation starts (from pending taps)
        or continues (a retained proposal the scheduler sent no drafts for).
        Both answer with a fused replay's first posterior id, the target's
        greedy choice after the row's token, in the flat int32 form the device
        sampler produces. Every other case is the plain decode.
        """

        def plain():
            return Gemma4ForCausalLM.decode_forward(
                self, *args, page_tables_per_layer=page_tables_per_layer, **self._dflash_plain_kwargs(kwargs)
            )

        if len(live) != 1:
            self._dflash_leave_speculation("several live rows", [self._dflash_slot_of(r, kwargs) for r in live])
            return plain()
        row = live[0]
        if kwargs.get("sampling_params") is None:
            # Host sampling reads logits; the fused replay yields ids only.
            slot = self._dflash_slot_of(row, kwargs)
            if self._dflash_owns(self._dflash_pending_owner, slot):
                self._dflash_drop_pending()
            if self._dflash_owns(self._dflash_live_owner, slot):
                self._dflash_end_session()
            return plain()
        token = self._dflash_solo_next_token(row, tokens, start_pos, rows, kwargs, page_tables_per_layer)
        if token is None:
            return plain()
        out = torch.zeros(int(rows), dtype=torch.int32)
        out[row] = int(token)
        return out

    def _dflash_solo_next_token(self, row, tokens, start_pos, rows, kwargs, page_tables_per_layer):
        """The one live row's next token from the drafter session, or None.

        None means the row cannot speculate on this step and the caller runs
        the plain decode; the session state is already cleaned up for that.
        """
        anchor = self._dflash_row(tokens, rows, row)[0]
        position = self._dflash_row(start_pos, rows, row)[0] if start_pos is not None else None
        slot = self._dflash_slot_of(row, kwargs)
        if self._spec_active:
            if not self._dflash_owns(self._dflash_live_owner, slot):
                # Another request's session; its owner is not in this step and
                # keeps it. This row's taps cannot bootstrap past it.
                self._dflash_drop_pending()
                return None
            retained = self._dflash_retained
            if (
                retained is not None
                and not retained.consumed
                and retained.anchor_token == anchor
                and (position is None or retained.anchor_position == position)
            ):
                # The scheduler sent no drafts for this proposal (it drops
                # them near max_model_len). The replay's first id is still
                # the answer, and the next propose commits one token.
                retained.consumed = True
                self._dflash_note_step(row, kwargs, page_tables_per_layer)
                return retained.posterior[0]
            self._dflash_end_session()
            return None
        if self._spec_pending is None:
            return None
        if not self._dflash_owns(self._dflash_pending_owner, slot):
            # Taps of a request that never decoded alone (finished, preempted).
            self._dflash_drop_pending()
            return None
        if row != 0 or position is None or not self._dflash_can_bootstrap(position, kwargs.get("page_table")):
            self._dflash_drop_pending()
            return None
        self._dflash_bootstrap(row, anchor, position, kwargs, page_tables_per_layer)
        _drafts, posterior = self._spec_decoder.contract_replay(first=self._spec_first_step)
        self._spec_first_step = False
        return int(posterior[0])

    def _dflash_can_bootstrap(self, position, page_table):
        dec = self._spec_decoder
        if dec is None or page_table is None or not self._spec_width_set:
            return False
        # ``_spec_bootstrap`` captures a per-session trace when no prepared
        # width covers the position. That capture writes through an all-zero
        # scratch page table, and under bounded sliding block 0 is a live ring
        # block, so this rail never reaches it.
        return self._dflash_decline_reason(dec, int(position)) is None

    def _dflash_bootstrap(self, row, anchor, position, kwargs, page_tables_per_layer):
        owner = self._dflash_pending_owner
        self._spec_bootstrap(
            int(anchor),
            int(position),
            kwargs.get("page_table"),
            kwargs.get("kv_cache"),
            page_tables_per_layer=page_tables_per_layer,
        )
        self._dflash_pending_owner = None
        self._dflash_live_owner = owner
        self._spec_owner_slot = owner[0] if owner else None
        self._dflash_retained = None
        # The bootstrap refreshed the verify page tables from this table.
        page_table = kwargs.get("page_table")
        self._dflash_last_pt = page_table[:1].clone() if page_table is not None else None
        self._dflash_note_step(row, kwargs, page_tables_per_layer)

    def _dflash_decline_reason(self, dec, position):
        """Why no verify block can be written at ``position``, or None.

        The block occupies rows ``position .. position + P_v - 1``: the anchor
        and K drafts. Each must be covered by a captured width and by
        ``max_seq_len``, and on an exact-window ring must not wrap onto a slot
        the anchor's own query still reads.
        """
        p_v = int(dec.P_v)
        if dec.width_for(position) is None:
            return f"no captured verify width covers position {position}"
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0) or 0)
        if max_seq_len and position + p_v > max_seq_len:
            return f"verify rows {position}..{position + p_v - 1} exceed max_seq_len {max_seq_len}"
        if self._dflash_ring is not None:
            ring, window = self._dflash_ring
            if ring - window < p_v - 1 and position + p_v > ring:
                return f"verify rows would wrap the exact-window ring {ring} (window {window}, P_v {p_v})"
        return None

    def _dflash_verify_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live, num_valid):
        """A step the runner marked speculative: a ``[rows, 1+K]`` candidate block.

        Returns ``VerifyOutput(spec_mode="argmax_ids")``. The row carrying
        drafts is answered from the retained proposal once the block matches
        it; a mismatch raises, because answering would report a posterior for
        tokens the device never evaluated. Rows without drafts take column 0
        from the plain decode, or, for the one live row, from the session as
        an ordinary step would.
        """
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, VerifyOutput

        width = self._SPEC_CONTRACT_K + 1
        valid = num_valid.reshape(-1).tolist() if num_valid is not None else [0] * int(rows)
        drafted = [r for r in live if r < len(valid) and int(valid[r]) > 0]
        ids = torch.full((int(rows), width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        if len(drafted) > 1:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: rows {drafted} carry drafts, but one drafter session exists"
            )
        if drafted:
            row = drafted[0]
            retained = self._dflash_take_retained(
                row, self._dflash_slot_of(row, kwargs), tokens, start_pos, rows, int(valid[row])
            )
            posterior = retained.posterior[:width]
            ids[row, : len(posterior)] = torch.tensor(posterior, dtype=torch.int32)
            if len(live) == 1:
                self._dflash_note_step(row, kwargs, page_tables_per_layer)
                return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
            # A peer joined after the proposal. The device evaluated the drafts
            # already, so the drafted row is answered and kept out of the
            # peers' plain decode; speculation ends with this step.
            plain = self._dflash_plain_argmax(args, kwargs, page_tables_per_layer, rows, exclude_row=row)
            for r in range(int(rows)):
                if r != row:
                    ids[r, 0] = plain[r]
            self._dflash_leave_speculation("peer joined while a proposal was outstanding")
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        if len(live) == 1:
            row = live[0]
            token = self._dflash_solo_next_token(row, tokens, start_pos, rows, kwargs, page_tables_per_layer)
            if token is not None:
                ids[row, 0] = int(token)
                return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        else:
            self._dflash_leave_speculation("several live rows", [self._dflash_slot_of(r, kwargs) for r in live])
        ids[:, 0] = self._dflash_plain_argmax(args, kwargs, page_tables_per_layer, rows)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)

    def _dflash_take_retained(self, row, slot, tokens, start_pos, rows, valid):
        """The retained proposal that answers ``row``'s block, marked consumed.

        ``slot`` is the state slot the row reads this step. Raises when that
        slot owns no session, when no unconsumed proposal is retained, or when
        the block's anchor, position or first ``valid`` draft columns differ
        from the proposal. Truncation to fewer drafts is fine: the posterior at
        column j depends only on columns before it.
        """
        block = self._dflash_row(tokens, rows, row)
        position = self._dflash_row(start_pos, rows, row)[0] if start_pos is not None else None
        where = f"row {row} (slot {slot}), position {position}, anchor {block[0]}"
        if not self._spec_active or not self._dflash_owns(self._dflash_live_owner, slot):
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                "but this row owns no drafter session"
            )
        retained = self._dflash_retained
        if retained is None or retained.consumed:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                "but no proposal is retained for them"
            )
        if retained.anchor_token != block[0] or (position is not None and retained.anchor_position != position):
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify block for {where} does not continue the "
                f"retained proposal (anchor {retained.anchor_token} at position {retained.anchor_position})"
            )
        if valid > len(retained.drafts) or valid > len(block) - 1:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                f"but the retained proposal has {len(retained.drafts)}"
            )
        sent = block[1 : 1 + valid]
        if sent != retained.drafts[:valid]:
            raise ValueError(
                "Gemma4DFlash speculative contract: verify was sent draft ids this model did not "
                f"propose: block={sent} proposed={retained.drafts[:valid]} ({where})"
            )
        retained.consumed = True
        return retained

    @staticmethod
    def _dflash_column_zero(args, kwargs, rows, exclude_row=None):
        """Narrow a ``[rows, 1+K]`` block to the plain decode's ``[rows, 1]`` tokens and ``[rows]`` positions.

        Column 0 is each row's last committed token at its own position, which
        is the plain decode's input; padding rows keep their -1 position.
        ``exclude_row`` is given position -1 too, so the target skips it.
        """
        args = list(args)
        kwargs = dict(kwargs)

        def col0(t):
            if t is None:
                return None
            v = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
            if int(rows) <= 0 or v.numel() % int(rows):
                return v
            return v.reshape(int(rows), -1)[:, 0].contiguous()

        def tokens(t):
            v = col0(t)
            return v if v is None else v.reshape(int(rows), 1)

        def positions(t):
            v = col0(t)
            if v is not None and exclude_row is not None and 0 <= int(exclude_row) < int(v.numel()):
                v = v.clone()
                v[int(exclude_row)] = -1
            return v

        for name, index, fn in (("tokens", 0, tokens), ("start_pos", 1, positions)):
            if kwargs.get(name) is not None:
                kwargs[name] = fn(kwargs[name])
            elif len(args) > index:
                args[index] = fn(args[index])
        return tuple(args), kwargs

    @staticmethod
    def _dflash_is_host(tt_out):
        """The runner's own test for a decode output that needs no device read."""
        if isinstance(tt_out, torch.Tensor):
            return True
        if isinstance(tt_out, tuple):
            return all(t is None or isinstance(t, torch.Tensor) for t in tt_out)
        return False

    def _dflash_convert(self, host_groups, rows, is_tokens):
        """Per-group host tensors to one torch tensor at this step's row count.

        ``Generator.process_decode_output_host`` converts at ``max_batch_size``
        rows, which reads a narrower host logits batch as ``max_batch_size``
        rows of a truncated vocabulary.
        """
        groups = list(host_groups) if isinstance(host_groups, (list, tuple)) else [host_groups]
        per = max(1, int(rows) // max(1, len(groups)))
        parts = []
        for model, group in zip(self.model, groups):
            tensor = group[0] if isinstance(group, tuple) else group
            parts.append(model.process_output_decode(tensor, per, S=1, is_tokens=is_tokens))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _dflash_plain_argmax(self, args, kwargs, page_tables_per_layer, rows, exclude_row=None):
        """Each row's next-token id from the plain decode, int32 ``[rows]``.

        Device-sampled ids are used as they are (greedy sampling is the
        argmax, and this rail is greedy); host logits are converted at this
        step's row count and argmaxed over the vocabulary.
        """
        args, kwargs = self._dflash_column_zero(args, kwargs, rows, exclude_row)
        kwargs = self._dflash_plain_kwargs(kwargs)
        is_tokens = kwargs.get("sampling_params") is not None
        tt_out = Gemma4ForCausalLM.decode_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        if exclude_row is not None:
            # The excluded row ran at position -1 and left a stale resident
            # position behind; its next plain decode must reload.
            self._dflash_mark_stale(exclude_row)
        if self._dflash_is_host(tt_out):
            host = tt_out[0] if isinstance(tt_out, tuple) else tt_out
        else:
            host = self._dflash_convert(
                Gemma4ForCausalLM.read_decode_output(self, tt_out, async_read=False), rows, is_tokens
            )
        host = host if isinstance(host, torch.Tensor) else torch.as_tensor(host)
        ids = host.reshape(-1) if is_tokens else host.reshape(-1, int(host.shape[-1])).argmax(dim=-1)
        out = torch.zeros(int(rows), dtype=torch.int32)
        n = min(int(rows), int(ids.numel()))
        out[:n] = ids.reshape(-1)[:n].to(torch.int32)
        return out

    # -- propose ---------------------------------------------------------------
    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        """``[rows, K]`` int32 drafts for the next step, drafted on device.

        Only the one live row of a solo step receives drafts, and only while
        it owns the live session. The runner's accepted count and anchor are
        committed first (the fused body merges the previous replay's rows at
        the start of the next one), the verify page tables are refreshed to
        the row's current block table, and the anchor position is checked
        against the captured widths, ``max_seq_len`` and the bounded ring
        before the replay. A decline returns ``num_valid`` 0 for every row and
        ends the session; the row continues as plain decode.
        """
        del hidden
        k = int(num_drafts)
        rows = int(committed.shape[0]) if committed is not None and committed.dim() > 1 else 1
        live = self._dflash_live_rows(positions, rows)
        if len(live) != 1:
            self._dflash_leave_speculation("several live rows at proposal", live)
            return self._dflash_decline(rows, k)
        row = live[0]
        if not self._spec_active or not self._dflash_owns(self._dflash_live_owner, row):
            return self._dflash_decline(rows, k)
        retained = self._dflash_retained
        if retained is not None and not retained.consumed:
            self._dflash_decline_session("the previous proposal was never answered")
            return self._dflash_decline(rows, k)
        dec = self._spec_decoder
        block = self._dflash_row(committed, rows, row)
        n = int(counts.reshape(-1)[row]) if counts is not None else 1
        n = max(1, min(n, len(block)))
        self._dflash_steps += 1
        self._dflash_committed += n
        anchor = block[n - 1]
        anchor_position = self._dflash_row(positions, rows, row)[n - 1] if positions is not None else None
        dec.contract_commit(n, anchor)
        position = int(dec.start)
        if anchor_position is not None and position != int(anchor_position):
            self._dflash_decline_session(
                f"decoder position {position} disagrees with the runner's anchor position {anchor_position}"
            )
            return self._dflash_decline(rows, k)
        self._dflash_refresh_tables(dec)
        reason = self._dflash_decline_reason(dec, position)
        if reason is not None:
            self._dflash_decline_session(reason)
            return self._dflash_decline(rows, k)
        dec.select_width(position)
        drafts, posterior = dec.contract_replay(first=self._spec_first_step)
        self._spec_first_step = False
        drafts = [int(t) for t in list(drafts)[:k]]
        self._dflash_retained = RetainedProposal(
            anchor_token=int(anchor),
            anchor_position=position,
            drafts=drafts,
            posterior=[int(t) for t in posterior],
        )
        out = torch.zeros((rows, k), dtype=torch.int32)
        out[row, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        valid[row] = len(drafts)
        return self._dflash_draft_output(out, valid)

    def _dflash_decline_session(self, reason):
        logger.info(f"Gemma4DFlash speculative contract: proposal declined, session ends ({reason})")
        self._dflash_end_session()

    def _dflash_refresh_tables(self, dec):
        """Point the persistent verify page tables at the owner's current blocks.

        The runner allocates a block roughly every ``block_size`` tokens. The
        verify page-table buffers keep what they held at bootstrap until
        refreshed, and a KV write past that table's last block lands in the
        null block. Under bounded sliding the per-layer tables are rebuilt and
        installed first, because ``refresh_page_tables`` reads the installed
        ring tables for the sliding layers.
        """
        tables = self._dflash_owner_tables
        if tables is None:
            return
        page_table, page_tables_per_layer, kv_cache = tables
        if page_table is None:
            return
        row = page_table[:1] if page_table.dim() > 1 else page_table.reshape(1, -1)
        previous = self._dflash_last_pt
        if previous is not None and previous.shape == row.shape and torch.equal(previous, row):
            return
        if self._bounded_sliding_kv_cache:
            per_layer = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
            per_layer = self._pad_sliding_page_tables_for_bounded(per_layer, kv_cache, authoritative=True)
            if per_layer:
                self.model[0]._active_page_tables_per_layer = per_layer
        dec.refresh_page_tables(row)
        self._dflash_last_pt = row.clone()

    @staticmethod
    def _dflash_draft_output(draft_token_ids, num_valid):
        """A ``DraftOutput``, carrying per-row counts when the plugin has them."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        if "num_valid" in getattr(DraftOutput, "__dataclass_fields__", {}):
            return DraftOutput(draft_token_ids=draft_token_ids, num_valid=num_valid)
        return DraftOutput(draft_token_ids=draft_token_ids)

    @classmethod
    def _dflash_decline(cls, rows, k):
        """A proposal that offers no row a draft.

        The ids are always ``[rows, K]`` in-vocabulary values, so the tensor
        cannot say "nothing this step"; ``num_valid`` 0 does, and the runner
        records nothing for such a row.
        """
        return cls._dflash_draft_output(
            torch.zeros((int(rows), int(k)), dtype=torch.int32),
            torch.zeros(int(rows), dtype=torch.int32),
        )


class Gemma4MTPForCausalLM(Gemma4ForCausalLM):
    """Gemma4 with the it-assistant (KV-shared) drafter, serving at B=1.

    Same session pattern as :class:`Gemma4DFlashForCausalLM` but no prefill
    taps (the drafter cross-attends the target's own KV), so prefill stays on
    the normal traced path. Each decode call is ONE fused draft+verify
    iteration via ``SpeculativeDecoder.serving_step``.
    """

    _SPEC_K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", "5").replace("auto", "5"))
    _SPEC_N = _SPEC_K + 1

    model_capabilities = {
        **Gemma4ForCausalLM.model_capabilities,
        "supports_async_decode": os.environ.get("GEMMA4_SPEC_ASYNC", "0") != "0",
        "supports_sample_on_device": True,
        "output_tokens_per_step": _SPEC_N,
        # ADAPTIVE block-output (mirrors the dFlash twin): emit the spec row only
        # when decoding ALONE (batch==1); batch>1 decodes as plain batched
        # baseline at width 1, for which the adaptive scheduler reserved exactly
        # one placeholder. This lets ONE server run max_num_seqs>1 -- MTP spec at
        # conc-1, baseline batched above it -- instead of pinning max_num_seqs=1.
        # Speculation only pays bandwidth-bound / low batch anyway.
        "tt_adaptive_block_output": _SPEC_N > 1,
        # One fused draft+verify iteration writes exactly the _SPEC_N rows it
        # emits, so here the physical extent EQUALS the emitted width and the
        # plugin's twice-the-width fallback already covers it. Declared anyway so
        # the bound is explicit rather than incidental: dFlash gets this wrong the
        # moment its block is narrower than its verify (vllm-tt-plugin#118 review,
        # finding 2), and MTP would too if a future step emitted less than it
        # verified.
        "tt_block_kv_extent_tokens": _SPEC_N if _SPEC_N > 1 else 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_assistant = None
        self._spec = None
        self._spec_pending = None  # prompt_len awaiting first decode
        self._spec_pending_owner = None  # request the pending session belongs to
        # Request a LIVE session belongs to: _spec_pending_owner covers only the
        # pre-bootstrap window, so without this a solo decode step scheduled for
        # a non-owner ran the block loop against the owner's session (see the
        # dFlash class's _spec_active_owner for the async path that reaches it).
        self._spec_active_owner = None
        self._spec_first_step = True
        self._spec_last_pt = None
        # Verify-width ladder for the live session (see mtp_pv_width_ladder).
        self._spec_width_ladder = None
        # Warmup-captured session: serving RESEEDS it instead of capturing per
        # request (see warmup_model_decode).
        self._mtp_warmup_capture = os.environ.get("GEMMA4_MTP_WARMUP_CAPTURE", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self._spec_warm = False
        if self._bounded_sliding_kv_cache:
            _reserve_spec_ring_headroom(getattr(self._text_config(), "sliding_window", None), self._SPEC_N, "Gemma4MTP")
        self._spec_horizon = int(os.environ.get("GEMMA4_MTP_SERVE_HORIZON", "2048"))
        logger.info(
            f"Gemma4MTP serving: K={self._SPEC_K} (N={self._SPEC_N}/step), "
            f"horizon={self._spec_horizon} new tokens/request, B=1 sessions"
        )

    def warmup_model_decode(self, *args, **kwargs):
        """Capture the verify-width ladder ONCE, here, instead of per request.

        Per-request capture does not survive concurrency. On a BH Galaxy DP=4
        box at max_concurrency 128 it produced 48 captures and 33.7 s of capture
        time INSIDE decode steps, and a worker missed vLLM's RPC deadline by
        minutes (``TimeoutError: RPC call to sample_tokens timed out``). The 31B
        dFlash twin passes the same sweep with 0 serving-time captures because
        its widths are captured in warmup; this gives MTP the same treatment.

        GEMMA4_MTP_WARMUP_CAPTURE=0 restores per-request capture.
        """
        self._decode_warmup_complete = True
        if not self._mtp_warmup_capture or not kwargs.get("enable_trace"):
            del args
            logger.info("Gemma4MTP: decode warmup capture disabled; sessions capture per request")
            return None
        # A MISSING DRAFTER is not a capture failure and must not be swallowed:
        # per-request capture would hit the same absent weights and kill the
        # engine on the first SOLO request, after the server had already
        # reported ready. That is how the bh_loudbox leg failed -- the batched
        # benchmark passed (baseline path, no drafter needed) and the
        # single-request coherence guard then got an EngineDeadError 500.
        # Fail here instead, while the server is still starting, with the same
        # actionable message the dFlash twin gives.
        if os.environ.get("GEMMA4_ASSISTANT_MODEL") is None:
            _assistant = _assistant_default_snapshot(os.environ.get("HF_MODEL", "google/gemma-4-31B-it"))
            if not os.path.isdir(str(_assistant)):
                raise RuntimeError(
                    "Gemma4MTP: assistant (drafter) snapshot not found -- resolved "
                    f"{_assistant!r}, which does not exist. The MTP path cannot serve "
                    "without it, and starting anyway would serve plain baseline until "
                    "the first solo request killed the engine. Set "
                    "GEMMA4_ASSISTANT_MODEL to a local snapshot, or make the weights "
                    "cache writable so google/gemma-4-<size>-it-assistant can be "
                    "fetched (CI: run with mlperf-read-only false once to populate it)."
                )
        try:
            self._spec_warmup_session(kwargs.get("kv_cache"), kwargs.get("num_blocks"))
        except Exception as e:
            # A warmup that cannot capture must not take the server down: fall
            # back to per-request capture and SAY which path is live, because
            # the two have very different behaviour under concurrency.
            logger.warning(
                f"Gemma4MTP: warmup width-set capture failed ({e!r}); falling back "
                "to per-request capture (expect capture stalls at concurrency)"
            )
            self._spec_release_session(force=True)
        return None

    def _spec_warmup_session(self, kv_cache, num_blocks):
        """Build the assistant + one persistent SpeculativeDecoder and capture
        every ladder width, before any request exists."""
        import time as _time

        from models.demos.gemma4.tt.common import create_assistant_model
        from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

        if kv_cache is None:
            raise ValueError("warmup capture needs the KV cache")
        model0 = self.model[0]
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0))
        if not max_seq_len:
            raise ValueError("warmup capture needs max_seq_len")
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        if self._spec_assistant is None:
            assistant_path = os.environ.get("GEMMA4_ASSISTANT_MODEL") or _assistant_default_snapshot(
                os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
            )
            # Sized for the WHOLE context, not one request's start: the cached
            # assistant used to be sized from the FIRST request's position, so a
            # later, longer request ran against a drafter whose KV could not
            # reach it.
            _, self._spec_assistant = create_assistant_model(
                mesh_device=self.mesh_device,
                target_model=model0,
                mesh_config=model0.mesh_config,
                ccl_manager=model0.ccl_manager,
                assistant_path=assistant_path,
                max_seq_len=max_seq_len + 64,
            )
        blocks = int(num_blocks) if num_blocks else max(1, max_seq_len // 64)
        scratch_pt = torch.zeros(1, blocks, dtype=torch.int32)
        t0 = _time.time()
        spec = SpeculativeDecoder(
            target_model=model0,
            assistant_model=self._spec_assistant,
            mesh_device=self.mesh_device,
            tt_kv_cache=kv_layers,
            page_table_torch=scratch_pt,
            stop_tokens=set(),
            draft_len=self._SPEC_K,
        )
        spec._use_trace = True
        ladder = mtp_pv_width_ladder(max_seq_len, self._SPEC_K)
        cost = spec.serving_warmup_widths(ladder)
        self._spec = spec
        self._spec_warm = True
        self._spec_width_ladder = ladder
        self._spec_budget_end = max(ladder) - self._SPEC_K - 2
        logger.info(
            f"Gemma4MTP: warmup captured {len(cost)} verify widths in "
            f"{_time.time()-t0:.1f}s (max_model_len={max_seq_len}, widths={ladder})"
        )

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        # ADAPTIVE: a spec session is B=1. A batched prefill is served as plain
        # baseline and arms NO session -- the decode side then serves those rows
        # through the batched fallback.
        if tokens is not None and int(tokens.shape[0]) != 1:
            self._spec_release_session()
            return super().prefill_forward(*args, **kwargs)
        out = super().prefill_forward(*args, **kwargs)
        prompt_lens = kwargs.get("prompt_lens")
        n = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        self._spec_pending = n
        self._spec_owner_slot = _spec_first_slot(kwargs.get("empty_slots"))
        # Record WHICH request these pending taps belong to. At max_num_seqs>1
        # several prefills can land before the next solo decode step, and the
        # session slot is global: without this the decode would bootstrap from
        # another prompt's length -- wrong tokens, not just a wrong width.
        self._spec_pending_owner = self._spec_pt_identity(kwargs.get("page_table"))
        return out

    def _spec_bootstrap(self, anchor_id, start, page_table, kv_cache, page_tables_per_layer=None):
        import time as _time

        from models.demos.gemma4.tt.common import create_assistant_model
        from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

        n = self._spec_pending
        self._spec_pending = None
        if start != n:
            logger.warning(f"Gemma4MTP: first decode start_pos {start} != prompt_len {n}")
        if self._spec is not None:
            self._spec.serving_release()
            self._spec = None
        model0 = self.model[0]
        # BOUNDED sliding target (auto-enabled at >=131072): install the hybrid
        # per-layer page tables BEFORE the SpeculativeDecoder capture so its
        # fused body reads the small RING pool for sliding layers and the flat
        # global table for full attention (spec_decode.py reads
        # ``_active_page_tables_per_layer`` throughout -- built for the metal
        # harness, which always installs this set). Without it the verify reads
        # the flat table for EVERY layer, sliding layers hit wrong physical
        # blocks, and acceptance collapses -- the same serving gap fixed for
        # dFlash. MTP is B=1 with a fresh capture per request and block-output
        # pre-allocates the full block table, so a one-time install per
        # bootstrap suffices (no per-block refresh needed).
        if self._bounded_sliding_kv_cache and page_table is not None:
            # No fallback: a missing install means the fused verify reads the
            # flat table for sliding layers and acceptance silently collapses
            # -- fail loudly instead (review finding on tt-metal#56048).
            if hasattr(model0, "_active_page_tables_per_layer"):
                del model0._active_page_tables_per_layer
            _ptpl = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
            _ptpl = self._pad_sliding_page_tables_for_bounded(_ptpl, kv_cache, authoritative=True)
            if not _ptpl:
                raise RuntimeError(
                    "Gemma4MTP: bounded per-layer page-table install produced "
                    "no tables; refusing to speculate against the flat table"
                )
            model0._active_page_tables_per_layer = _ptpl
        if self._spec_assistant is None:
            assistant_path = os.environ.get("GEMMA4_ASSISTANT_MODEL") or _assistant_default_snapshot(
                os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
            )
            _, self._spec_assistant = create_assistant_model(
                mesh_device=self.mesh_device,
                target_model=model0,
                mesh_config=model0.mesh_config,
                ccl_manager=model0.ccl_manager,
                assistant_path=assistant_path,
                max_seq_len=int(start) + self._spec_horizon + 64,
            )
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        t0 = _time.time()
        if self._spec_warm and self._spec is not None:
            # RESEED the warmup-captured session: no capture on the serving
            # path, which is what keeps a step inside vLLM's worker RPC deadline
            # at concurrency (see warmup_model_decode).
            self._spec.refresh_page_tables(page_table[:1] if page_table is not None else None)
            w = self._spec.serving_reseed(int(anchor_id), int(start))
            self._spec_active_owner = self._spec_pt_identity(page_table)
            self._spec_cur = (int(anchor_id), int(start))
            self._spec_first_step = False  # the reseed already staged this iteration
            self._spec_last_pt = page_table[:1].clone() if page_table is not None else None
            self._spec_budget_end = max(self._spec_width_ladder or [w]) - self._SPEC_K - 2
            logger.info(
                f"Gemma4MTP session: warm reseed {_time.time()-t0:.2f}s "
                f"(anchor={int(anchor_id)}, start={start}, width={w})"
            )
            return
        if self._spec is not None:
            self._spec.serving_release()
            self._spec = None
        spec = SpeculativeDecoder(
            target_model=model0,
            assistant_model=self._spec_assistant,
            mesh_device=self.mesh_device,
            tt_kv_cache=kv_layers,
            page_table_torch=page_table[:1] if page_table is not None else None,
            stop_tokens=set(),
            draft_len=self._SPEC_K,
        )
        spec._use_trace = True
        # WIDTH SET: a single capture sized to anchor + horizon makes the
        # horizon a hard generation budget -- past it this class used to end the
        # request with a full end-of-sequence row (review finding on
        # tt-metal#56048, same shape as step 3 for dFlash). The ladder lets the
        # session migrate to a wider trace instead. Only the rung this prompt
        # needs is captured now, so TTFT is unchanged; the rest are captured on
        # first crossing. Packed verify only -- the batch-dim verify has no
        # width-dependent capture, so it never had the cap.
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0))
        ladder = None
        if max_seq_len and spec._fused_packed_enabled():
            ladder = mtp_pv_width_ladder(max_seq_len, self._SPEC_K)
        if ladder:
            spec.serving_setup_widths(int(anchor_id), int(start), ladder)
        else:
            spec.serving_setup(int(anchor_id), int(start), max_new_tokens=self._spec_horizon)
        self._spec = spec
        self._spec_width_ladder = ladder
        self._spec_active_owner = self._spec_pt_identity(page_table)
        self._spec_cur = (int(anchor_id), int(start))
        self._spec_first_step = True
        # The capture bound THIS table; re-stage only when it changes after it.
        self._spec_last_pt = page_table[:1].clone() if page_table is not None else None
        # With a ladder the reach is the WIDEST rung, not the horizon: the
        # session migrates rather than stopping, so what remains here is a
        # backstop at the end of the context (vLLM's own max_model_len stop
        # arrives first). Without a ladder the horizon is still the budget.
        self._spec_budget_end = (
            max(ladder) - self._SPEC_K - 2 if ladder else int(start) + self._spec_horizon - self._SPEC_N - 1
        )
        logger.info(
            f"Gemma4MTP session: seed+capture {_time.time()-t0:.1f}s " f"(anchor={int(anchor_id)}, start={start})"
        )

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        start_pos = kwargs.get("start_pos")
        if start_pos is None and len(args) > 1:
            start_pos = args[1]
        if tokens is None:
            raise ValueError("Gemma4MTP decode expects token input")
        # Adaptive block-output: a BATCHED decode step (concurrency>1) runs plain
        # baseline and returns the RAW device output, which the runner's
        # read_decode_output/process_decode_output_host pipeline converts, trims
        # to the real batch and samples exactly as for a baseline model. A solo
        # request that just joined a batch drops its MTP session first, so
        # baseline owns its KV from vLLM's committed position.
        if self._spec_real_batch(tokens, start_pos) != 1:
            self._spec_release_session()
            return super().decode_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        anchor_from_runner = int(tokens.reshape(-1)[0])
        if self._spec_pending is not None and not self._spec_pending_is_mine(kwargs.get("page_table")):
            # The pending session was captured for a DIFFERENT request (its
            # owner finished or was aborted before it ever decoded). Bootstrap-
            # ing it here would speculate from another prompt's length. Drop it
            # and serve this request as plain baseline -- the width the adaptive
            # scheduler reserved for a non-owner row.
            logger.warning(
                "Gemma4MTP: pending spec session belongs to another request; serving this one as plain baseline"
            )
            self._spec_release_session()
        if self._spec_pending is not None:
            start = int(start_pos.reshape(-1)[0]) if start_pos is not None else None
            self._spec_bootstrap(
                anchor_from_runner,
                start,
                kwargs.get("page_table"),
                kwargs.get("kv_cache"),
                page_tables_per_layer=page_tables_per_layer,
            )
        if self._spec is not None and not self._spec_active_is_mine(kwargs.get("page_table")):
            # Solo decode step for a request that does NOT own the live session
            # (the owner can be skipped by upstream's num_output_placeholders
            # guard once it reaches max_tokens, leaving this one alone with the
            # session still armed). Running the block loop here would emit the
            # owner's speculation for this request against a single reserved
            # placeholder.
            logger.warning(
                "Gemma4MTP: live spec session belongs to another request; "
                "releasing it and serving this one as plain baseline"
            )
            self._spec_release_session()
        if self._spec is None or self._spec_cur is None:
            # No session for this row (batched prefill, or a dropped non-owner
            # session): serve plain baseline, matching the reserved width.
            #
            # _spec and _spec_cur are DIFFERENT things: _spec is the capture,
            # _spec_cur is the active request's (token, position). A warm
            # release keeps the capture -- the warmup traces are a process
            # artifact, not a per-session one, and freeing them would leave no
            # captured widths and no way to recapture -- and nulls _spec_cur
            # only. Gating on _spec alone therefore let an ordinary batch-to-solo
            # transition fall through to `cur_token, cur_pos = self._spec_cur`
            # and raise TypeError. Nothing below can arm it: the pending
            # bootstrap is above, so a None here means no request owns the
            # capture.
            return super().decode_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        # Re-stage the fused verify's page tables when vLLM's block table for
        # this request CHANGES. The KV manager allocates a block only every
        # ~block_size tokens, so a session captured at prefill holds the
        # prompt's blocks and zeros past them: once generation crosses out of
        # the prompt's last block the verify reads and WRITES the null block.
        # The dFlash twin has always done this (refresh_page_tables); MTP
        # claimed a one-time install sufficed because "block-output
        # pre-allocates the full block table", which is not what the runner
        # hands over -- the table's WIDTH is fixed at max_num_blocks_per_req,
        # its CONTENT grows. Same finding as tt-metal#55548 D2 on the qwen36
        # MTP verify trace, and here it also reaches the host map behind the
        # per-iteration hot-block uploads (see refresh_page_tables).
        cur_pt = kwargs.get("page_table")
        if cur_pt is not None:
            row = cur_pt[:1] if cur_pt.dim() > 1 else cur_pt
            prev = getattr(self, "_spec_last_pt", None)
            if prev is None or not torch.equal(prev, row):
                if self._bounded_sliding_kv_cache:
                    # Sliding layers ring on a static pool; the FULL-attention
                    # table is the one that grows, so rebuild the hybrid set
                    # from what the runner passed this step before re-staging.
                    _ptpl = self._build_per_layer_page_tables(page_tables_per_layer, cur_pt)
                    _ptpl = self._pad_sliding_page_tables_for_bounded(_ptpl, kwargs.get("kv_cache"), authoritative=True)
                    if _ptpl:
                        self.model[0]._active_page_tables_per_layer = _ptpl
                self._spec.refresh_page_tables(row)
                self._spec_last_pt = row.clone()
        cur_token, cur_pos = self._spec_cur
        if cur_pos >= self._spec_budget_end:
            logger.warning(
                f"Gemma4MTP: horizon ({self._spec_horizon} new tokens) exhausted at "
                f"position {cur_pos}; ending the request with EOS"
            )
            return torch.full((1, self._SPEC_N), self._eos_fill_id(), dtype=torch.int32)
        # BLOCK LOOP: one MTP iteration commits an accepted prefix + bonus, which
        # is VARIABLE (1..N) -- acceptance is content-dependent. Run iterations
        # back-to-back until the step's row holds N tokens, exactly as the dFlash
        # twin fills _SPEC_BLOCK.
        #
        # Without this loop a normal low-acceptance iteration returns a short row
        # that has to be padded, and any pad value is wrong: -1 puts an invalid
        # id on the wire, and EOS terminates the request at the first pad because
        # the scheduler trims at the first stop token. That is not hypothetical --
        # it capped solo generation at ~one block (osl=128 returned 6 tokens)
        # while the batched baseline path, which never pads, ran to full length.
        eos_id = self._eos_fill_id()
        base_pos = cur_pos
        block = []
        while len(block) < self._SPEC_N:
            if cur_pos >= self._spec_budget_end:
                # Horizon exhausted mid-row: EOS here is a GENUINE stop.
                block.append(eos_id)
                break
            committed, m = self._spec.serving_step(cur_token, cur_pos)
            cur_pos += m + 1
            cur_token = committed[-1]
            block.extend(committed)
            if eos_id in committed:
                break
        block = block[: self._SPEC_N]
        # Resume from exactly what was EMITTED, not from how far the iterations
        # ran: a final iteration may overshoot the row. The next step re-drafts
        # those positions and overwrites them (the implicit-overwrite state
        # rollback the plugin contract relies on), so the emitted stream and the
        # session position stay in lockstep.
        self._spec_cur = (block[-1], base_pos + len(block))
        # A row shorter than N now means a genuine stop only, so EOS-filling the
        # tail is correct: upstream trims at the first stop token.
        out = torch.full((1, self._SPEC_N), eos_id, dtype=torch.int32)
        out[0, : len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    def read_decode_output(self, tt_out, async_read=False, *_, **__):
        # A SOLO SPEC step returns committed host tokens (a torch.Tensor) from
        # decode_forward -- nothing to read, pass them straight through (no
        # events). The ADAPTIVE BATCHED fallback returns the RAW DEVICE output
        # of the baseline decode instead, so route that to the base reader
        # (events under async_read) for the deferred read pipeline. Mirrors the
        # dFlash twin; without this the device tensor reaches the runner as if
        # it were host tokens and the engine dies on the first batched step.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *_, **__)

    def _eos_fill_id(self) -> int:
        """Token used to fill a short spec row. See decode_forward."""
        eos = getattr(self.model[0].hf_config, "eos_token_id", 1)
        eos_set = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
        return int(min(eos_set))

    def _spec_pending_is_mine(self, page_table) -> bool:
        """True when the pending session was captured for the request whose page
        table this is. Unknown identity on either side falls back to True (the
        pre-existing single-session behaviour)."""
        owner = getattr(self, "_spec_pending_owner", None)
        cur = self._spec_pt_identity(page_table)
        if owner is None or cur is None:
            return True
        return owner == cur

    def _spec_active_is_mine(self, page_table) -> bool:
        """True when the LIVE session belongs to the request whose page table
        this is. Unknown identity on either side falls back to True, matching
        _spec_pending_is_mine."""
        owner = getattr(self, "_spec_active_owner", None)
        cur = self._spec_pt_identity(page_table)
        if owner is None or cur is None:
            return True
        return owner == cur

    def _spec_release_session(self, force: bool = False) -> None:
        """Drop any pending/active MTP session (batched fallback and lifecycle).

        Also drops the bounded per-layer page tables this session installed, so a
        later BATCHED baseline decode (the adaptive fallback) rebuilds its own set
        instead of reading this request's stale B=1 ring tables. ``_spec_bootstrap``
        deletes and re-installs them itself, so it is unaffected -- it calls
        ``serving_release`` directly rather than going through here.
        """
        self._spec_pending = None
        self._spec_pending_owner = None
        self._spec_active_owner = None
        self._spec_last_pt = None
        if self._spec_warm and not force:
            # The warmup-captured traces are a warmup artifact, not a
            # per-session capture: freeing them would leave the process with no
            # captured widths and no way to recapture (warmup is over) -- i.e.
            # back to the per-request capture this exists to remove. End the
            # SESSION only; the per-layer tables still go below.
            self._spec_cur = None
        else:
            self._spec_width_ladder = None
            if self._spec is not None:
                self._spec.serving_release()
                self._spec = None
        try:
            if hasattr(self.model[0], "_active_page_tables_per_layer"):
                del self.model[0]._active_page_tables_per_layer
        except Exception:
            pass

    # -- plugin lifecycle hooks (block-output contract) -----------------------
    def note_state_slots_moved(self, moves) -> None:
        """The runner gathered per-slot state: ``moves`` is slot ``old -> new``.

        The B=1 session's owner slot is the identity ``release_request`` compares
        against, and the runner permutes slots between steps, so it has to follow
        the move or a release stops matching its own request. The runner passes
        the whole permutation at once because applying pairs one at a time can
        move the same owner twice (vllm-tt-plugin#118 review, finding 1).
        """
        owner = getattr(self, "_spec_owner_slot", None)
        if owner is None or not moves:
            return
        try:
            moved = moves.get(int(owner))
        except (AttributeError, TypeError, ValueError):
            return
        if moved is not None:
            self._spec_owner_slot = int(moved)

    def release_request(self, row: int) -> None:
        owner_slot = getattr(self, "_spec_owner_slot", None)
        if owner_slot is not None and row is not None and int(row) != int(owner_slot):
            # ANOTHER request's slot: adaptive serving admits several live
            # requests while this adapter keeps one session, so releasing here
            # would strand a live owner on baseline width (see the dFlash twin).
            return
        self._spec_owner_slot = None
        self._spec_release_session()

    def release_persistent_capture(self) -> None:
        # NOT force: the warm branch keeps the warmup-captured widths on
        # purpose, because warmup is over and nothing could recapture them.
        self._spec_owner_slot = None
        try:
            self._spec_release_session()
        finally:
            super().release_persistent_capture()
