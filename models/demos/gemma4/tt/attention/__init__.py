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

import os

from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, Mode

from .weights import AttentionWeights, load_attention_weights
from .kv_cache import init_kv_cache
from .decode import decode_forward, packed_decode_forward
from .prefill import flush_deferred_bounded_fills, prefill_forward


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

        # Persistent hot-block staging for the packed-verify loop-free KV write.
        # Allocated lazily by the spec-decode driver (see tt/spec_decode.py);
        # None means packed_decode_forward falls back to the per-position loop.
        self.kv_staging = None

        # Trace-safe cross-chunk sliding-tail pool. The stash's runtime-allocated
        # clones can land on a live prefill trace's baked scratch, and an
        # INTERLEAVED request's trace replay then clobbers them (#30187 class —
        # proven by put/get checksums: fluent nondeterministic corruption of the
        # victim's continuation at conc>=2 with 1024-bucket remnants). Boot-time
        # allocation reserves addresses no later capture can alias; tails are
        # ttnn.copy'd in at stash time and cloned out (transient, same-call) at
        # consume time.
        # GEMMA4_TAIL_POOL_SLOTS bounds concurrent chunked-prefill requests per
        # sliding layer (evict-oldest beyond it). The pool is boot-DRAM-resident
        # on every sliding layer, so memory-starved parts (WH T3K: 12 GB/chip
        # cannot boot 31B with 8 slots) can shrink it; 0 disables the pool and
        # falls back to the legacy runtime stash, which is NOT trace-safe under
        # interleaved replay (the G8 clobber) — use only for bring-up.
        self._tail_pool = None
        self._tail_pool_map = {}
        _pool_slots = max(0, int(os.environ.get("GEMMA4_TAIL_POOL_SLOTS", "8")))
        if config.is_sliding and config.sliding_window and _pool_slots == 0 and layer_idx == 0:
            logger.warning(
                "GEMMA4_TAIL_POOL_SLOTS=0: cross-chunk sliding tails use the "
                "runtime clone stash, which is NOT trace-safe under interleaved "
                "replay (#30187 class) — bring-up only, do not serve with this."
            )
        if config.is_sliding and config.sliding_window and _pool_slots:
            tp = max(1, int(getattr(mesh_config, "tp", 1)))
            nkv_local = 1 if self.weights.kv_replicated else max(1, config.num_key_value_heads // tp)
            shape = [1, nkv_local, int(config.sliding_window), int(config.head_dim)]
            self._tail_pool = [
                (
                    ttnn.zeros(
                        shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.zeros(
                        shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
                for _ in range(_pool_slots)
            ]

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
            # Cross-chunk sliding tails are PER REQUEST: under vLLM, chunks of
            # different requests interleave through this layer in scheduler
            # order (and rows are re-ordered between rounds — plugin PR #68's
            # "mutable live-batch indexing" lesson), so a single per-layer slot
            # hands one request's window tail to another request's continuation
            # (fluent nondeterministic corruption of the victim's final chunk,
            # measured at conc3/9k). Key the stash by the request's stable
            # identity — its first global block id — set by the generator via
            # ``config._g4_active_req_key``; None (demo / non-vLLM paths) keys a
            # single legacy slot.
            _req_key = getattr(self.config, "_g4_active_req_key", None)
            if isinstance(chunk_start_idx, ttnn.Tensor):
                pass
            elif chunk_start_idx is None or int(chunk_start_idx) == 0:
                # First chunk of THIS request: drop only this request's stale
                # tail — releasing the shared slot here wiped tails other
                # requests still needed (second victim mode of the same bug).
                self._release_sliding_prefill_tail(req_key=_req_key)
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
                sliding_tail_in=self._get_sliding_tail(_req_key),
            )
            # prefill_forward consumed (deallocated) the incoming tail; stash the
            # new one for the next chunk under this request's key.
            self._put_sliding_tail(_req_key, sliding_tail_out)
            self._last_kv = kept_kv
            return tt_out

    # ── per-request sliding-tail stash ─────────────────────────────────────
    _SLIDING_TAIL_MAX_KEYS = 33  # max concurrent requests (32) + legacy None slot

    def _get_sliding_tail(self, req_key):
        # Pool path: clone out of the boot-allocated (trace-safe) slot. The
        # transient clones are consumed within this same forward call, before
        # any other request's replay can run.
        pool_map = getattr(self, "_tail_pool_map", None) or {}
        slot = pool_map.get(req_key) if req_key else None
        if slot is not None and self._tail_pool is not None:
            kb, vb = self._tail_pool[slot]
            return (
                ttnn.clone(kb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.clone(vb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            )
        tails = getattr(self, "_sliding_tails_by_key", None)
        if tails is None:
            return None
        entry = tails.get(req_key)
        if entry is not None:
            tails[req_key] = None  # consumed by prefill_forward (it deallocates)
        return entry

    def _put_sliding_tail(self, req_key, tail):
        if tail is None:
            return
        pool = self._tail_pool
        # Pool only for real runtime requests (truthy key): warmup/traced paths
        # must keep the legacy stash — a ttnn.copy during capture TT_FATALs
        # ("Cannot load new binaries during trace capture").
        if pool is not None and req_key:
            k, v = tail
            hist = int(self.config.sliding_window)
            if int(k.shape[-2]) != hist:
                from .prefill import _left_pad_kv_to_hist

                k, v = _left_pad_kv_to_hist(k, v, hist, self.config.head_dim, deallocate_inputs=True)
            kb_shape = list(pool[0][0].shape)
            if list(k.shape) == kb_shape and list(v.shape) == kb_shape:
                pool_map = self._tail_pool_map
                slot = pool_map.pop(req_key, None)
                if slot is None:
                    used = set(pool_map.values())
                    free = [i for i in range(len(pool)) if i not in used]
                    if free:
                        slot = free[0]
                    else:
                        # Evict oldest — but spill its tail into the legacy
                        # stash first: dropping it would make the evicted
                        # request's next chunk attend without its previous
                        # window (silently wrong). The spilled clone carries
                        # the G8 trace-clobber risk only for that request,
                        # only while chunked concurrency exceeds the pool.
                        oldest_key = next(iter(pool_map))
                        slot = pool_map.pop(oldest_key)
                        okb, ovb = pool[slot]
                        spill = (
                            ttnn.clone(okb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                            ttnn.clone(ovb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        )
                        tails = getattr(self, "_sliding_tails_by_key", None)
                        if tails is None:
                            tails = {}
                            self._sliding_tails_by_key = tails
                        self._dealloc_tail(tails.pop(oldest_key, None))
                        tails[oldest_key] = spill
                pool_map[req_key] = slot
                kb, vb = pool[slot]
                ttnn.copy(k, kb)
                ttnn.copy(v, vb)
                for t in (k, v):
                    try:
                        t.deallocate(True)
                    except Exception:
                        pass
                return
            tail = (k, v)  # shape mismatch — fall back to the clone stash
        tails = getattr(self, "_sliding_tails_by_key", None)
        if tails is None:
            tails = {}
            self._sliding_tails_by_key = tails
        old = tails.pop(req_key, None)
        self._dealloc_tail(old)
        tails[req_key] = tail
        while len(tails) > self._SLIDING_TAIL_MAX_KEYS:
            evict_key = next(iter(tails))
            self._dealloc_tail(tails.pop(evict_key))

    def _dealloc_tail(self, tail):
        if not tail:
            return
        persistent = getattr(self.config, "sliding_prefill_tail_persistent", None)
        if persistent is not None and len(tail) == 2 and len(persistent) == 2 and tail[0] is persistent[0]:
            return  # persistent ring buffers are owned by the traced path
        for t in tail:
            try:
                t.deallocate(True)
            except Exception:
                pass

    def _release_sliding_prefill_tail(self, *, clear_persistent: bool = False, req_key=..., all_keys: bool = False):
        """Drop cross-chunk sliding tails.

        Default (``req_key`` given): drop only that request's tail — the shared
        single-slot release wiped tails other in-flight requests still needed.
        ``all_keys`` / no ``req_key``: drop every stashed tail (generator-level
        clears around trace capture). Traced multi-chunk binds persistent K/V
        ring buffers into the captured graph. Soft release keeps those buffers
        so runtime replay can ``ttnn.copy`` into the same addresses. Hard clear
        (``clear_persistent``) is only for sp0 compile↔capture: both passes must
        take the first-alloc path — leaving persistent set makes capture hit
        ``ttnn.copy`` without that program in cache (TT_FATAL
        !is_capturing_trace, WH-T3K nightly).
        """
        tails = getattr(self, "_sliding_tails_by_key", None) or {}
        pool_map = getattr(self, "_tail_pool_map", None)
        if req_key is not ... and not all_keys and not clear_persistent:
            self._dealloc_tail(tails.pop(req_key, None))
            if pool_map is not None:
                pool_map.pop(req_key, None)  # pool buffers persist (boot-owned)
            return
        persistent = getattr(self.config, "sliding_prefill_tail_persistent", None)
        if clear_persistent:
            seen: set[int] = set()
            groups = list(tails.values()) + ([persistent] if persistent is not None else [])
            for group in groups:
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
            tails.clear()
            if pool_map is not None:
                pool_map.clear()  # pool buffers persist (boot-owned); mappings must not
            self.config.sliding_prefill_tail_persistent = None
            return
        for key in list(tails.keys()):
            self._dealloc_tail(tails.pop(key))
        if pool_map is not None:
            pool_map.clear()  # pool buffers persist (boot-owned)
