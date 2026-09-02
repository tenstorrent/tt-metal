# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Attention module.

Uses HF-style ttnn.experimental.rotary_embedding — no Meta-format weight conversion,
no transformation matrices. Cos/sin caches are passed directly.

Supports two layer types:
- sliding_attention: head_dim=256, 8 KV heads, separate K/V, full RoPE, window=1024
- full_attention: head_dim=512, 2 KV heads, K=V tying, partial RoPE (0.25), full context
"""

import ttnn
from models.demos.gemma4.config import MeshConfig, Mode
from models.demos.gemma4.tt.ccl import cp_degree

from .weights import AttentionWeights, load_attention_weights
from .kv_cache import init_kv_cache
from .decode import decode_forward, packed_decode_forward
from .operations import build_cp_prefill_mask
from .prefill import flush_deferred_bounded_fills, prefill_forward
from .ring_prefill import init_packed_ring_kv_cache, init_ring_kv_cache


class Gemma4AttentionConfig:
    """Configuration for a single attention layer, derived from HF config + layer type."""

    def __init__(self, hf_config, layer_idx):
        self.layer_type = hf_config.layer_types[layer_idx]
        self.hidden_size = hf_config.hidden_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.rms_norm_eps = hf_config.rms_norm_eps
        # Propagated for weight-load policy (e.g. skip DRAM-shard on MoE for PCC).
        self.enable_moe_block = bool(getattr(hf_config, "enable_moe_block", False))

        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_kv_tying = getattr(hf_config, "attention_k_eq_v", False) and not self.is_sliding

        if self.is_sliding:
            self.num_key_value_heads = hf_config.num_key_value_heads
            self.head_dim = hf_config.head_dim
            self.sliding_window = hf_config.sliding_window
            self.rope_theta = hf_config.rope_theta
            self.partial_rotary_factor = 1.0
        else:
            # Global KV heads: use num_global_key_value_heads if set, else fall back to sliding
            global_kv = getattr(hf_config, "num_global_key_value_heads", None)
            self.num_key_value_heads = global_kv if global_kv else hf_config.num_key_value_heads
            self.head_dim = getattr(hf_config, "global_head_dim", hf_config.head_dim)
            self.sliding_window = None
            self.rope_theta = hf_config.global_rope_theta
            self.partial_rotary_factor = hf_config.partial_rotary_factor

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # When set (only on sliding-window layers wired with bounded allocations),
        # the three paged ops (paged_fill_cache / paged_update_cache /
        # paged_scaled_dot_product_attention_decode) wrap the absolute position
        # into a circular buffer of this many tokens before the page_table lookup.
        # Mirrors vLLM's SlidingWindowSpec: physical cache holds only
        # cache_position_modulo / block_size blocks per sequence; the per-layer
        # page_table is zero-padded out to max_model_len / block_size.
        self.cache_position_modulo = None


class Gemma4Attention:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        layer_idx,
        tensor_cache_path=None,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=131072,
        weight_dtype=ttnn.bfloat16,
        bounded_sliding_kv_cache: bool = False,
        ring_prefill_chunk_size=None,
        # Legacy parameter — ignored (no longer needed with HF-style RoPE)
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx

        # vLLM-style hybrid kv_cache_groups: SlidingWindowSpec layers allocate only
        # sliding_window/block_size blocks per sequence and pass cache_position_modulo
        # to the three paged ops, which wrap absolute positions into the bounded slots.
        # Full-attention layers leave cache_position_modulo unset and take the legacy
        # unbounded path. Setting the field here is harmless when paged mode is off:
        # the call sites only read it inside their ``if page_table is not None`` branch.
        self.bounded_sliding_kv_cache = (
            bounded_sliding_kv_cache and config.is_sliding and config.sliding_window is not None
        )
        if self.bounded_sliding_kv_cache:
            config.cache_position_modulo = config.sliding_window

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            tensor_cache_path=tensor_cache_path,
            weight_dtype=weight_dtype,
        )

        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                tensor_cache_path=tensor_cache_path,
            )
        else:
            self.kv_cache = None

        # Ring cache for cross-chunk prefill under context parallelism. Contiguous and
        # CP-sharded along the sequence, which is what ring_joint reads (it takes the
        # cache directly, with no page table). Allocated only when multi-chunk CP
        # prefill is actually in play; the paged cache above is untouched.
        self.ring_kv_cache = None
        self.ring_max_seq_len = None
        # Deliberately NOT gated on create_kv_cache. Gemma4Model builds the paged
        # cache itself after constructing the layer and assigns it to
        # ``self_attn.kv_cache``, so this constructor always sees create_kv_cache
        # False on the model path — gating on it left the ring cache unallocated and
        # every cross-chunk read silently fell back to the mask path.
        if cp_degree(mesh_config) > 1 and ring_prefill_chunk_size:
            num_local_kv_heads = 1 if self.weights.kv_replicated else config.num_key_value_heads // mesh_config.tp
            if self.weights.is_global:
                self.ring_kv_cache = init_packed_ring_kv_cache(
                    mesh_device=mesh_device,
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            else:
                self.ring_kv_cache = init_ring_kv_cache(
                    mesh_device=mesh_device,
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    head_dim=config.head_dim,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            self.ring_max_seq_len = max_seq_len

        # Fallback CP mask cache for callers that pass no ccl_manager; the shared
        # one on CCLManager is preferred so a 60-layer stack holds two masks, not 60.
        self._cp_mask_cache_local = {}

        # Persistent hot-block staging for the packed-verify loop-free KV write.
        # Allocated lazily by the spec-decode driver (see tt/spec_decode.py);
        # None means packed_decode_forward falls back to the per-position loop.
        self.kv_staging = None

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        token_index=None,
        shared_kv=None,
        keep_kv=False,
        is_kv_shared=False,
        position_idx_cache=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
        sequential_kv_write=False,
        rope_presliced=False,
        packed=None,
        chunk_start_idx=None,
        chunk_page_table=None,
        packed_global_rope=None,
        packed_sliding_rope=None,
    ):
        """
        Attention forward pass — dispatches to on-device decode or prefill.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: (cos_cache, sin_cache) TT tensors, shape [1, 1, max_seq_len, head_dim]
            position_idx: position tensor for KV cache update (decode only)
            page_table: paged attention page table
            kv_cache: [k_cache, v_cache] or None
            is_decode: True for decode mode
            token_index: int position for decode RoPE slicing (decode only)
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)
            packed: optional packed-verify dict (decode only) — keys packed_p,
                position_idx, kv_write_idxs, attn_mask, rope_packed, embed_idx,
                hot_pt; routes to packed_decode_forward (P positions, one pass)
        """
        cache = kv_cache or self.kv_cache
        cos_cache, sin_cache = rope_mats

        # Do NOT release the sliding prefill tail on decode. Under vLLM
        # ``async_scheduling``, another request's decode can interleave between
        # APC continuation prefills of a different request; wiping the single
        # per-layer stash here drops ``sliding_tail_in`` for ``chunk_start>0``
        # (shield QB2: ``chunk_start=384 without sliding_tail_in``). Tails are
        # released when a new prefill starts at ``chunk_start_idx==0`` (below)
        # or when the generator explicitly clears them around trace capture.

        if is_decode and packed is not None:
            return packed_decode_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                position_idx=packed["position_idx"],
                kv_write_idxs=packed.get("kv_write_idxs"),
                attn_mask=packed["attn_mask"],
                packed_p=packed["packed_p"],
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                is_kv_shared=is_kv_shared,
                rope_packed=packed.get("rope_packed"),
                kv_staging=self.kv_staging,
                embed_idx=packed.get("embed_idx"),
                hot_pt=packed.get("hot_pt"),
            )

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                position_idx=position_idx,
                token_index=token_index,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                is_kv_shared=is_kv_shared,
                position_idx_cache=position_idx_cache,
                sequential_kv_write=sequential_kv_write,
                rope_presliced=rope_presliced,
            )
        else:
            # Sliding-window layers under generator-level chunked prefill carry a
            # rolling K/V window tail across chunks (stored on this per-layer
            # instance). Reset it at the start of a prefill (single-chunk, or the
            # first generator chunk with chunk_start_idx==0). Traced multi-chunk
            # passes a device tensor offset — the generator resets tails before
            # the first chunk; do not int()-cast the tensor here.
            if isinstance(chunk_start_idx, ttnn.Tensor):
                pass
            elif chunk_start_idx is None or int(chunk_start_idx) == 0:
                self._release_sliding_prefill_tail()
            tt_out, kept_kv, sliding_tail_out = prefill_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
                batch_size=batch_size,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
                chunk_start_idx=chunk_start_idx,
                chunk_page_table=chunk_page_table,
                sliding_tail_in=getattr(self, "_sliding_prefill_tail", None),
                cp_attn_mask=self._cp_attn_mask(hidden_states.shape[-2]),
                ring_kv_cache=self.ring_kv_cache,
                ring_max_seq_len=self.ring_max_seq_len,
                packed_global_rope=packed_global_rope,
                packed_sliding_rope=packed_sliding_rope,
            )
            # prefill_forward consumed (deallocated) the incoming tail; stash the
            # new one for the next chunk.
            self._sliding_prefill_tail = sliding_tail_out
            self._last_kv = kept_kv
            return tt_out

    def _cp_attn_mask(self, local_seq_len):
        """CP-sharded additive prefill mask for ``local_seq_len`` query rows.

        ``None`` when context parallelism is off, which leaves the existing
        ``is_causal`` / ``sliding_window_size`` SDPA path untouched.

        Cached per local sequence length: the build does host work, so it must
        happen during the warmup pass and not again inside a trace capture, where
        the mask has to be the same persistent device tensor every replay.
        """
        if cp_degree(self.mesh_config) <= 1:
            return None
        window = self.config.sliding_window if self.config.is_sliding else None
        # Shared across layers when a ccl_manager is available: the mask depends
        # only on (local_seq_len, window), so a 60-layer stack needs two entries.
        cache = getattr(self.ccl_manager, "_cp_mask_cache", None)
        if cache is None:
            cache = self._cp_mask_cache_local
        key = (local_seq_len, window)
        mask = cache.get(key)
        if mask is None:
            mask = build_cp_prefill_mask(self.mesh_device, self.mesh_config, local_seq_len, window)
            cache[key] = mask
        return mask

    def _release_sliding_prefill_tail(self, *, clear_persistent: bool = False):
        """Drop the cross-chunk sliding tail for the next prefill's first chunk.

        Traced multi-chunk binds persistent K/V ring buffers into the captured
        graph. Soft release (default) keeps those buffers so runtime replay can
        ``ttnn.copy`` into the same addresses. Hard clear (``clear_persistent``)
        is only for sp0 compile↔capture: both passes must take the first-alloc
        path — leaving persistent set makes capture hit ``ttnn.copy`` without
        that program in cache (TT_FATAL !is_capturing_trace, WH-T3K nightly).
        """
        tail = getattr(self, "_sliding_prefill_tail", None)
        persistent = getattr(self.config, "sliding_prefill_tail_persistent", None)
        if clear_persistent:
            seen: set[int] = set()
            for group in (tail, persistent):
                if group is None:
                    continue
                for t in group:
                    tid = id(t)
                    if tid in seen:
                        continue
                    seen.add(tid)
                    try:
                        t.deallocate(True)
                    except Exception:
                        pass
            self._sliding_prefill_tail = None
            self.config.sliding_prefill_tail_persistent = None
            return
        if tail is not None:
            is_persistent = (
                persistent is not None and len(tail) == 2 and len(persistent) == 2 and tail[0] is persistent[0]
            )
            if not is_persistent:
                for t in tail:
                    try:
                        t.deallocate(True)
                    except Exception:
                        pass
        self._sliding_prefill_tail = None
