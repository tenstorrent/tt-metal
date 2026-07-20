# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3.2 / GLM DSA lightning indexer (top-k key selection for sparse MLA).

A self-contained component owned by ttMLA. It owns the indexer-only state (weights, device
index-key cache, RoPE tables, arch constants) and runs the on-device stems / RoPE / collectives
/ logits / top-k. The shared q_a latent (qr) is passed into forward() by ttMLA; everything else it
reuses from the MLA layer — the SP×TP mesh + axes, compute-kernel configs, softmax scale,
weight-cache location and the CCL handles — is injected through the constructor; the indexer holds
no reference back to ttMLA (and no MLA weights) and runs its own TP/SP collectives. Inert for dense
v3.1 (no indexer weights → ttMLA never builds it).
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_indexer_key_chunk
from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_perm_matrix

# DSA indexer weight names are owned by TtIndexer.WEIGHT_NAMES (single source of truth). A
# module-level INDEXER_WEIGHT_NAMES alias is defined at the bottom of this file for back-compat.


class TtIndexer:
    """DSA lightning indexer for one MLA layer. Self-contained: owns the indexer weights, the
    grown-by-concat device index-key cache and the indexer RoPE tables, and runs its own TP/SP
    collectives. All MLA-layer dependencies it reuses are injected at construction (no ttMLA ref)."""

    # --- DSA ownership: weight names, config-field detection, host/cache API. ttMLA routes all
    # indexer weight-name / config-field / cache-file / placeholder decisions through these so the
    # DSA facts live here, not in ttMLA. Mirrors dense ttMLA's check/build/convert cache pattern.
    WEIGHT_NAMES = (
        "indexer.wq_b",
        "indexer.wk",
        "indexer.k_norm",
        "indexer.k_norm_bias",
        "indexer.weights_proj",
    )
    # Config fields that mark a runtime config as DSA-sparse (index_rope_interleave is optional and
    # defaults to False; the three below are the discriminator vs a dense DeepSeek-V3 / R1 config).
    REQUIRED_CONFIG_FIELDS = ("index_topk", "index_n_heads", "index_head_dim")

    @classmethod
    def matches_config(cls, config) -> bool:
        """True iff the runtime config carries the DSA indexer fields (dense R1/V3 lacks them)."""
        return all(getattr(config, name, None) is not None for name in cls.REQUIRED_CONFIG_FIELDS)

    @classmethod
    def has_host_weights(cls, state_dict) -> bool:
        """True iff a live state dict carries all indexer host tensors (from-weights callers)."""
        return bool(state_dict) and all(f"{n}.weight" in state_dict for n in cls.WEIGHT_NAMES)

    @classmethod
    def extract_host_weights(cls, state_dict) -> dict:
        """Non-mutating pull of the indexer host tensors out of a state dict (keyed by WEIGHT_NAMES)."""
        return {n: state_dict[f"{n}.weight"] for n in cls.WEIGHT_NAMES if f"{n}.weight" in state_dict}

    @staticmethod
    def _cache_short_name(weight_name: str) -> str:
        # wq_b's replicated layout uses a distinct cache stem so the pre-replication TP-sharded
        # tensorbin cannot satisfy completeness checks or be loaded by cache-only construction.
        return "wq_b_repl" if weight_name == "indexer.wq_b" else weight_name.split(".")[-1]

    @classmethod
    def check_cache_complete(cls, cache_path, cache_name_prefix: str) -> bool:
        """True iff every indexer tensorbin exists under cache_name_prefix (e.g. 'layer_0.mla').
        Uses a direct ``Path.glob`` (no `init_checker`/global-state dependency) because this also runs
        at ttMLA construction time — the resolver / __init__ gate — where the global fast-cache checker
        is not necessarily initialized. It's a one-off per-layer check (5 files), so the batch
        fast-checker optimization isn't needed here. Indexer files are `{prefix}.indexer_{short}` — a
        disjoint prefix space from the dense MLA names, so dense and indexer checks never alias."""
        if cache_path is None:
            return False
        cache_path = Path(cache_path)
        for name in cls.WEIGHT_NAMES:
            short = cls._cache_short_name(name)
            if not any(cache_path.glob(f"{cache_name_prefix}.indexer_{short}*.tensorbin")):
                logger.debug(f"TTNN indexer cache missing: {cache_name_prefix}.indexer_{short}")
                return False
        return True

    @classmethod
    def build_ttnn_cache(
        cls, idx_host, cache_path, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1
    ) -> None:
        """Write the indexer tensorbins to disk (device=None, no device copy)."""
        cls._convert_and_cache_weights(
            idx_host, mesh_device, config, layer_idx, sp_axis, tp_axis, cache_path=cache_path, device=None
        )

    @classmethod
    def _convert_and_cache_weights(
        cls, idx_host, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1, cache_path=None, device=None
    ):
        """Indexer weights → device (or cache). Mirrors dense MLA's converter:
        - host tensors present: transpose/shard/replicate and (optionally) write the cache;
        - `idx_host` falsy + `device=mesh_device`: build `torch.empty()` placeholders in the host
          (pre-transpose) shapes and rely on existing tensorbins (`as_tensor` ignores the placeholder
          on a cache hit);
        - `device=None`: build the cache only, return None.
        Returns the device-tensor dict keyed by short name (wq_b/wk/weights_proj/k_norm/k_norm_bias),
        or None when device is None. Cache filenames stay byte-compatible with the previously
        opportunistic `layer_{i}.mla.indexer_*` files (same dtype/layout/mapper)."""
        # A sparse config must carry the indexer geometry; assert loudly rather than silently defaulting
        # to garbage shapes.
        _missing = [f for f in ("index_n_heads", "index_head_dim") if not hasattr(config, f)]
        assert not _missing, f"indexer weight conversion requires config field(s) {_missing}"
        index_n_heads = config.index_n_heads
        index_head_dim = config.index_head_dim
        q_lora_rank = config.q_lora_rank
        hidden_size = config.hidden_size

        def _cache_name(short):
            return str(cache_path / f"layer_{layer_idx}.mla.indexer_{short}") if cache_path else None

        # A device load with no host weights must be backed by a complete tensorbin set, else
        # `as_tensor` converts the empty placeholders into garbage indexer weights. Mirror dense MLA's
        # lenient placeholder load (don't block construction) — but, unlike dense which is silent, WARN
        # loudly so the misuse is visible. The layer still stays sparse (binds TtIndexer); it does not
        # fall back to dense. (Build mode, device=None, is gated upstream by ttMLA.build_ttnn_cache.)
        if not idx_host and device is not None and not cls.check_cache_complete(cache_path, f"layer_{layer_idx}.mla"):
            logger.warning(
                f"Sparse MLA layer {layer_idx}: indexer has neither host weights nor a complete cache at "
                f"{cache_path!r}; loading from empty placeholders — indexer output will be garbage. "
                f"Build the indexer cache or pass the indexer weights."
            )

        if idx_host:
            wq_b = idx_host["indexer.wq_b"]
            wk = idx_host["indexer.wk"]
            wproj = idx_host["indexer.weights_proj"]
            knorm = idx_host["indexer.k_norm"]
            knorm_b = idx_host["indexer.k_norm_bias"]
        else:  # cache-only: placeholders in host (pre-transpose) shapes; as_tensor ignores them on a hit
            wq_b = torch.empty(index_n_heads * index_head_dim, q_lora_rank)
            wk = torch.empty(index_head_dim, hidden_size)
            wproj = torch.empty(index_n_heads, hidden_size)
            knorm = torch.empty(index_head_dim)
            knorm_b = torch.empty(index_head_dim)

        mem = ttnn.DRAM_MEMORY_CONFIG if device else None

        def repl(t, short, transpose=False):  # replicate across TP (transpose=True: host [out,in] -> device [in,out])
            return ttnn.as_tensor(
                (t.T if transpose else t).contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=_cache_name(short),
            )

        def shard(t, axis, short):  # host [out, in] -> device [in, out], dim `axis` sharded across tp
            dims = [None, None]
            dims[tp_axis] = axis
            return ttnn.as_tensor(
                t.T.contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mem,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
                cache_file_name=_cache_name(short),
            )

        # wq_b is REPLICATED (all H_idx heads on every chip) so the indexer_score head-sum is COMPLETE
        # on-chip — no TP logit all-reduce. (Was col-parallel/TP-head-sharded; replicating trades a small
        # matmul/score compute bump for dropping the ~end_pos-wide 2-CCL logit all-reduce.) Cache name
        # "wq_b_repl" (not "wq_b") so a stale col-sharded tensorbin can never alias this layout. wk /
        # weights_proj contract over hidden (TP-sharded) → upload transposed+sharded, reduced by _tp_rs_ag.
        result = {
            "wq_b": repl(
                wq_b, cls._cache_short_name("indexer.wq_b"), transpose=True
            ),  # [q_lora_rank, H_idx*D_idx] replicated (all heads)
            "wk": shard(wk, 0, "wk"),  # [dim, D_idx] sharded on dim
            "weights_proj": shard(wproj, 0, "weights_proj"),  # [dim, H_idx] sharded on dim
            "k_norm": repl(knorm, "k_norm"),  # [D_idx]
            "k_norm_bias": repl(knorm_b, "k_norm_bias"),  # [D_idx]
        }
        if device is None:
            for v in result.values():
                del v
            return None
        return result

    def __init__(
        self,
        idx_host,
        *,
        config,
        mesh_device,
        sp_axis: int,
        tp_axis: int,
        default_compute_kernel_config,
        hifi4_fp32_compute_kernel_config,
        weight_cache_path,
        layer_idx: int,
        tt_ccl,
        ccl_num_links: int,
        ccl_topology,
        seq_len: int = 1024,
        slot_num: int = 1,
        layer_num: int = 1,
    ):
        """Architecture constants are read from the HF config with no defaults (index_n_heads,
        index_head_dim, index_topk, index_rope_interleave — a sparse config that omits any of them
        fails loudly). θ / YaRN / rope table length come from the same config — single source of
        truth. Device index-key cache is grown by concat per chunk.

        Injected from ttMLA (the indexer keeps no back-reference): the SP×TP mesh + axes,
        compute-kernel configs, weight-cache location, and the CCL handles used by the inlined
        collectives (_tp_rs_ag / _sp_all_gather). The indexer derives its own softmax scale
        (index_head_dim**-0.5) — it does NOT reuse MLA's qk_head_dim*mscale scale. The q_a latent
        (qr) is passed into forward(), not held here — so the indexer holds no MLA weights."""
        self.config = config
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        mesh_shape = list(mesh_device.shape)
        self.sp_factor = mesh_shape[sp_axis]
        self.tp_factor = mesh_shape[tp_axis]
        self.default_compute_kernel_config = default_compute_kernel_config
        self.hifi4_fp32_compute_kernel_config = hifi4_fp32_compute_kernel_config
        # The q·kᵀ score is the indexer's only compute-bound op, so run it at LoFi; the projections keep
        # default_compute_kernel_config (HiFi2) — DRAM-bound, so lower fidelity buys nothing there.
        # indexer_score_dsa honors only math_fidelity (fp32_dest_acc_en must stay False — already the default).
        self._score_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
        )
        self.weight_cache_path = weight_cache_path
        self.layer_idx = layer_idx
        # Total local layers in the shared key cache. The block-cyclic index_kv_cache is user-major
        # [num_users*layer_num, 1, T, D_idx] — the SAME layout as the MLA KVPE cache — so the flat slot
        # for (user, layer) is cache_user_id*layer_num + cache_layer_idx, where cache_layer_idx is the
        # LOCAL per-rank cache slot passed to forward (mirrors the KVPE cache; NOT self.layer_idx, which is
        # GLOBAL and diverges from the local slot under pipeline parallelism). Computed in forward, used by
        # write_k and _gather_index_kbuf.
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl
        self.ccl_num_links = ccl_num_links
        self.ccl_topology = ccl_topology
        # Indexer geometry comes from the config with no defaults: a sparse config that omits any of these
        # fields fails loudly here rather than silently binding a wrong-shaped indexer.
        _required = ("index_n_heads", "index_head_dim", "index_topk", "index_rope_interleave")
        _missing = [f for f in _required if not hasattr(config, f)]
        assert not _missing, f"sparse MLA config is missing indexer field(s) {_missing}; must define all of {_required}"
        self.index_args = SimpleNamespace(
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            index_rope_interleave=config.index_rope_interleave,
        )
        self.seq_len = seq_len
        # Block-cyclic key-cache path: mirrors the MLA KVPE cache — a persistent, per-user/layer,
        # block-cyclic ND-sharded key cache written by update_padded_kv_cache and scored by
        # indexer_score_dsa's block-cyclic reader. The rope is always the interleaved on-device INDEXED op
        # (rotary_embedding_indexed). GLM is natively interleaved; DS-v3.2's half-split rope is reconciled
        # by _rope_perm below (permute q/k rope halves so the interleaved op matches the DS reference).
        # ALWAYS block-cyclic: single-shot is folded onto this path as one full-seq chunk at start_pos=0
        # (numerically identical to natural order — the block-cyclic reorder degenerates to a contiguous
        # per-chip SP shard), so the key cache persists layer-stacked and migrates to decode. The MLA that
        # owns this indexer feeds it the indexed rope tables and a caller-allocated index_kv_cache in both
        # modes.
        # Block-cyclic key cache (persistent [num_users*_index_cache_layers,1,S/sp,D_idx]) is NOT owned
        # here: exactly like the MLA KVPE cache, the caller allocates it and passes it into
        # forward(index_kv_cache=...) every call; the indexer never self-allocates it. write_k typecasts the
        # roped key to the cache's dtype before the in-place write, so the caller controls the dtype (BF8
        # validated — rotated + chunked suites match BF16 within bf16 noise, ~5e-4 PCC — so it can be
        # allocated BF8 to halve mem).
        # GLM-5.2 cross-layer indexer reuse: the index key cache is allocated for full layers only, so this
        # layer writes/reads its compacted rank among full layers, and the folded (user-major) slot stride
        # is num_full, not all layers. _index_cache_layers is that stride.
        self._num_index_layers = num_full_indexer_layers(config)
        self._is_index_compact = self._num_index_layers is not None
        self._index_layer_idx = full_indexer_rank(config, layer_idx) if self._is_index_compact else layer_idx
        self._index_cache_layers = self._num_index_layers if self._is_index_compact else self.layer_num
        self._upload_weights(idx_host)
        # DS block-cyclic uses the interleaved rotary_embedding_indexed op, but DS weights emit the
        # half-split (rotate_half) rope arrangement. Permute the rope half (half-split -> interleaved) so
        # the interleaved op pairs the right dims with each frequency; applied to BOTH q and k, the
        # permutation cancels in q·k, so the score (hence top-k) matches the DS half-split reference. GLM
        # is natively interleaved -> no permute. NOTE: the stored key is then in interleaved layout —
        # reindex by rope.interleaved_to_halfsplit_perm to compare it against a half-split reference.
        self._rope_perm = None
        if not self.index_args.index_rope_interleave:
            # rope.interleaved_perm_matrix owns the half-split -> interleaved convention (single source).
            self._rope_perm = ttnn.from_torch(
                interleaved_perm_matrix(64).to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

    # Inlined TP/SP collectives — the indexer owns its own copy so it depends on tt_ccl, not on ttMLA
    # (the dense MLA forward keeps its own equivalents; both go through the same tt_ccl handles).
    def _tp_rs_ag(self, t, rs_only=False):
        """All-reduce over TP = reduce-scatter (dim 3) then all-gather; rs_only stops after the RS."""
        if self.tp_factor == 1:
            return t
        t = ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )
        if rs_only:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )

    def _sp_all_gather(self, t, dim):
        """All-gather across the SP axis (sequence) → full-S replicated on SP. sp=1: no-op."""
        if self.sp_factor == 1:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.sp_axis,
        )

    def _tp_all_gather(self, t, dim):
        """All-gather across the TP axis → the TP-seq-shards reassembled to the SP block's full rows,
        replicated on TP. tp=1: no-op. (Spike helper for TP×SP query parallelism: regathers the top-k
        indices that were computed on TP-seq-sharded query rows back to the [1,1,S/sp,k] contract.)"""
        if self.tp_factor == 1:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )

    def _upload_weights(self, idx_host):
        """Indexer weights → device via the shared converter. `idx_host` may be a full host dict
        (from-weights) or falsy (cache-only: the converter builds placeholders and reads the
        `layer_{idx}.mla.indexer_*` tensorbins). The converter also writes the cache opportunistically
        on a from-weights load, exactly as before."""
        w = self._convert_and_cache_weights(
            idx_host,
            self.mesh_device,
            self.config,
            self.layer_idx,
            self.sp_axis,
            self.tp_axis,
            cache_path=self.weight_cache_path,
            device=self.mesh_device,
        )
        self._idx_wq_b = w["wq_b"]
        self._idx_wk = w["wk"]
        self._idx_wproj = w["weights_proj"]
        self._idx_knorm_w = w["k_norm"]
        self._idx_knorm_b = w["k_norm_bias"]

    def _bc_rope_pe(self, x: ttnn.Tensor, rope_tensors: dict, kv_actual_global: int) -> ttnn.Tensor:
        """Block-cyclic INDEXED RoPE on the rope half (first 64) of the last dim (block-cyclic path only).
        x [1, n_heads, S/sp, D_idx] SP-sharded on seq; rope_tensors are the whole-cache block-cyclic
        cos/sin/trans built by RotarySetup.get_rope_tensors_indexed — reused verbatim from ttMLA (the
        interleaved table, same 64-dim, shared with the MLA q_pe/k_pe rope). The op derives each shard-row's
        block-cyclic global position on-device from kv_actual_global, exactly as MLA's _apply_rope_padded,
        so keys land at the same positions update_padded_kv_cache writes them to. For DS (half-split
        weights) self._rope_perm first reorders the rope half into the interleaved arrangement so this
        interleaved op matches the DS reference (the permutation cancels in q·k, applied to both q and k)."""
        h, n = x.shape[1], x.shape[2]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, n, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, n, self.index_args.index_head_dim])
        if self._rope_perm is not None:  # DS: half-split -> interleaved arrangement for the interleaved op
            pe_i = ttnn.linear(pe, self._rope_perm, compute_kernel_config=self.hifi4_fp32_compute_kernel_config)
            ttnn.deallocate(pe)
            pe = pe_i
        pe = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            pe,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=kv_actual_global,
            cluster_axis=self.sp_axis,
        )
        out = ttnn.concat([pe, nope], dim=-1)
        ttnn.deallocate(pe)
        ttnn.deallocate(nope)
        return out

    def _gather_index_kbuf(self, index_kbuf: ttnn.Tensor, cache_batch_idx: int) -> ttnn.Tensor:
        """Read the block-cyclic ND-sharded key cache back to a replicated full-T [1,1,T,D_idx]
        (block-cyclic order preserved, bf16 TILE) for indexer_score_dsa's block-cyclic reader — the
        analogue of ttMLA._gather_kvpe_prefix, and it uses the same fix.

        SLOT SELECT BEFORE THE GATHER: index_kbuf is user-major [num_users*layer_num, 1, T, D_idx]
        (same layout as the MLA KVPE cache). Slice the active (user, layer) slot out of dim 0 FIRST, then
        SP all-gather only that single [1,1,T,D_idx] slot — NOT the whole B-slot cache. Gathering all
        slots materializes a full-T copy of every user/layer (OOMs at high num_layers, exactly like the
        MLA kvpe gather did). The gathered kv is then batch-1, so indexer_score needs NO cache_batch_idx
        (the op requires kB==1 when cache_batch_idx is unset). The unwritten suffix is never scored
        (future positions are causally masked).

        PERF TODO: this SP all-gather is currently a blocking barrier — it materializes the whole full-T
        key cache before indexer_score_dsa runs. It should instead be FUSED INTO the score op (ring-joint
        style, like ring_mla / ring-joint SDPA fuse the KV all-gather with the attention compute): pipeline
        the per-slab gather with the score matmul so each SP key slab is gathered and scored as it arrives,
        overlapping the CCL with the op's own compute instead of paying a full gather up front. Op-level
        change (ring indexer_score), not a host-side reorder."""
        cache_i = ttnn.to_memory_config(index_kbuf, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        if cache_i.shape[0] > 1:  # user-major slot select BEFORE the gather (single-slot cache → skip)
            sel = ttnn.slice(
                cache_i,
                [cache_batch_idx, 0, 0, 0],
                [cache_batch_idx + 1, 1, cache_i.shape[2], cache_i.shape[3]],
            )
            ttnn.deallocate(cache_i)
            cache_i = sel
        full = self._sp_all_gather(cache_i, dim=2)  # [1,1,T,D_idx] replicated, block-cyclic
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        return full

    def write_k(
        self, hidden_states, seq_len, start_pos, rope_tensors=None, cache_user_id=0, cache_layer_idx=0, index_kbuf=None
    ):
        """Device K stem (wk + TP all-reduce + k_norm + device rope) written into the device index-key
        cache. forward() calls this on every chunk so the key-cache stays complete — else later chunks
        score against missing keys for the early prefix. (Dense v3.1 binds a NullIndexer, so write_k never
        runs there.) Always block-cyclic (single-shot is folded onto it as one full-seq chunk at
        start_pos=0): rope the PER-CHIP shard at its block-cyclic positions, then write it in place via
        update_padded_kv_cache (per-(user,layer) slot, pad-aware kv_actual_global offset) — no SP
        all-gather, no O(n^2) concat; the cache stays SP-sharded."""
        k = ttnn.linear(
            hidden_states,
            self._idx_wk,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # per-chip partial [1, 1, S/sp, D_idx]
        k = self._tp_rs_ag(k)  # all-reduce over TP
        k = ttnn.layer_norm(
            k,
            weight=self._idx_knorm_w,
            bias=self._idx_knorm_b,
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # Rope the per-chip shard at its block-cyclic positions (kv_actual_global=start_pos), then write it
        # into this (user, layer) slot. update_padded_kv_cache places each chip's rows at the block-cyclic
        # offset (pad-aware) — the same math the query/key rope above uses. Single-shot is folded onto this
        # path as one full-seq chunk at start_pos=0, so the indexer is always block-cyclic. num_layers is the
        # compacted stride (_index_cache_layers) so it matches the cache_batch_idx computed in forward().
        k = self._bc_rope_pe(k, rope_tensors, start_pos)  # [1, 1, S/sp, D_idx] bf16
        if k.dtype != index_kbuf.dtype:  # write dtype must match the cache (update_padded_kv_cache asserts)
            k = ttnn.typecast(k, index_kbuf.dtype)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            index_kbuf,
            k,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self._index_cache_layers,
            kv_actual_global=start_pos,
            cluster_axis=self.sp_axis,
        )
        ttnn.deallocate(k)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        qr: ttnn.Tensor,
        seq_len: int,
        start_pos: int = 0,
        rope_tensors: dict = None,
        cache_user_id: int = 0,
        cache_layer_idx: int = 0,
        index_kv_cache: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Indexer forward → top-k key indices [1, 1, S/sp, k] over the device index-key cache, SP-sharded
        on the query axis (each chip scores its own S/sp rows; no Q/W all-gather). Fully on-device:
        stems, RoPE, cache, logits, topk — no host.

        ``index_kv_cache``: the persistent per-user key cache, allocated by the caller and passed in every
        call — the same ownership as ttMLA's KVPE ``kvpe_cache``. ALWAYS required (the indexer never
        self-allocates it): the indexer is always block-cyclic, and single-shot is folded onto that path as
        one full-seq chunk at offset 0, so there is no natural path that skips the cache.

        ``qr`` is the shared q_a latent (q_a_proj + TP all-reduce + q_a_layernorm) — ttMLA computes it once
        and passes it in; the indexer applies wq_b to it (no q_a stem of its own). ``qr`` is NOT deallocated
        here — ttMLA's _q_stem consumes it afterwards. ``hidden_states`` is still needed for the K stem
        (write_k) and the per-head weights (weights_proj). (write_k is called internally here; ttMLA.forward
        only ever calls self._indexer.forward — it never calls write_k directly.)

        ``rope_tensors`` (the MLA's block-cyclic indexed cos/sin/trans) and ``cache_user_id`` (per-user slot)
        drive the per-user block-cyclic key cache + block-cyclic scoring. Scoring currently spans the full
        preallocated cache width (see the kv_len TODO below)."""
        a = self.index_args
        if self._is_index_compact:
            cache_layer_idx = self._index_layer_idx
        glob = seq_len * self.sp_factor  # global query/key count this chunk
        end_pos = start_pos + glob
        # Block-cyclic key cache is caller-owned (like the KVPE cache) — required, never self-allocated.
        assert index_kv_cache is not None, (
            "block-cyclic indexer requires an externally-allocated index_kv_cache passed to forward() "
            "(same ownership as the MLA KVPE cache); none was provided"
        )
        # Flat user-major slot into the shared [num_users*_index_cache_layers, 1, T, D_idx] cache — same
        # formula as ttMLA._cache_batch_idx for the KVPE cache (cache_layer_idx is the LOCAL per-rank cache
        # slot, compacted to the full-layer rank above for GLM-5.2 cross-layer reuse). Written by write_k
        # and sliced by _gather_index_kbuf.
        cache_batch_idx = cache_user_id * self._index_cache_layers + cache_layer_idx
        self.write_k(
            hidden_states,
            seq_len,
            start_pos,
            rope_tensors=rope_tensors,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            index_kbuf=index_kv_cache,
        )

        # Q stem: the shared q_a latent (qr) -> indexer wq_b.
        q = ttnn.linear(
            qr,
            self._idx_wq_b,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, S/sp, H_idx*D_idx] — ALL heads (wq_b replicated); queries stay SP-sharded (rotation-safe)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=a.index_n_heads, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, H_idx, S/sp, D_idx] — all heads resident
        # block-cyclic indexed rope (same op/tables as the key rope + MLA q_pe)
        q_dev = self._bc_rope_pe(q, rope_tensors, start_pos)

        # weights_proj: device stem -> FULL all-reduce over tp (all H_idx heads, matching the replicated
        # wq_b heads) -> scale -> [1, 1, S/sp, H_idx].
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wts = self._tp_rs_ag(wts)  # full all-reduce (RS+AG) over tp -> all H_idx head-weights, replicated
        # Indexer softmax scale = index_head_dim**-0.5 (NO mscale), matching the reference IndexerCPU
        # (model.py: softmax_scale = head_dim**-0.5). Distinct from MLA's qk_head_dim*mscale**2 scale —
        # though as a uniform positive multiplier it cannot change the top-k selection regardless.
        wts = ttnn.multiply(wts, a.index_n_heads**-0.5 * a.index_head_dim**-0.5)  # [1,1,S/sp,H_idx] repl on tp

        # indexer_score wants per-head weights [1, H_idx, S/sp, 1]; wts is [1, 1, S/sp, H_idx].
        weights = ttnn.permute(wts, (0, 3, 2, 1))

        # TP×SP query parallelism (rope-then-split). q_dev/weights were roped on the FULL S/sp slab
        # (block-cyclic-correct, cluster_axis=sp_axis), so every row already carries its true position; now
        # split those rows over TP so each chip scores only S/(sp·tp) of them — indexer_score + topk shrink
        # ~TP×. RoPE is per-row so the split is safe (no 2-D rope op needed). The score is told the TP axis via
        # seq_shard_axes below, so its EXACT block-cyclic geometry adds each device's tp_rank*Sq' sub-offset
        # (rotation-safe). topk runs on the sub-rows; indices are all-gathered back over TP to the [1,1,S/sp,k]
        # contract so mla.py / sparse_sdpa are unchanged (both DeepSeek and GLM ride this one path).
        tpsp = self.tp_factor > 1
        if tpsp:
            q_full, weights_full = (
                q_dev,
                weights,
            )  # release the full-S slabs once TP-split (mesh_partition allocates new)
            q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=self.tp_axis)  # [1,H_idx,S/(sp·tp),D_idx]
            weights = ttnn.mesh_partition(weights, dim=2, cluster_axis=self.tp_axis)  # [1,H_idx,S/(sp·tp),1]
            ttnn.deallocate(q_full)
            ttnn.deallocate(weights_full)
            sq_local = seq_len // self.tp_factor
            qc = 64 if sq_local % 64 == 0 else 32  # q_chunk must divide the per-chip query tile count
        else:
            qc = 64
        # Causality is fused inside indexer_score (future columns -> -inf from chunk_start_idx), so no triu
        # mask here. All H_idx heads are resident on-chip (wq_b replicated), so head_group_size=0 reads the
        # key cache ONCE — but that needs L1 headroom, so k_chunk is bounded by resident head count
        # (DSA_INDEXER_CONFIG, measured per model: DeepSeek@64h=64, GLM@32h=224; larger OOMs).
        k_chunk = get_indexer_key_chunk(a.index_n_heads)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=qc, k_chunk_size=min(k_chunk, end_pos), head_group_size=0)
        # SP-sharded queries (rotation-safe): each chip scores its S/sp rows vs the full block-cyclic key
        # cache, with a per-device causal offset from cluster_axis=sp_axis (chip r: chunk_start = start_pos
        # + r*Sq, Sq=S/sp=seq_len). All H_idx heads on-chip -> the logit is COMPLETE (no partial-logit
        # all-reduce). topk stays SP-sharded ([1,1,S/sp,k]) -> fed straight to sparse_mla.
        # _gather_index_kbuf slices this (user, layer) slot then gathers it to replicated full-T; the op
        # reads it back in logical order via invP (batch-1, so cache_batch_idx=None) and applies the
        # straddle for a non-slab-aligned start_pos (padded chunk). Scores the FULL preallocated width T:
        # positions past each query are causally -inf, and top-k below drops those (-inf -> sentinel).
        #
        # Bound the score to the real written prefix (kv_len=end_pos) rather than the full preallocated
        # width T: end_pos = start_pos + chunk_global is the tightest legal value (the pad query rows
        # push the fullest-device causal window to end_pos; the op TT_FATALs on kv_len < that). kv_len
        # only WRITES logits[:, :, :, :end_pos] and leaves the tail [end_pos, T) STALE (not -inf); the
        # top-k below is told the valid length (valid_length=end_pos) so it never reads or ranks that
        # stale tail — which is the future top-k would drop anyway (causally -inf), so the selection is
        # unchanged.
        k_full = self._gather_index_kbuf(index_kv_cache, cache_batch_idx)  # [1,1,T,D_idx] bf16 TILE, block-cyclic
        logits = ttnn.experimental.indexer_score_dsa(
            q_dev,
            k_full,
            weights,
            chunk_start_idx=start_pos,
            program_config=cfg,
            compute_kernel_config=self._score_compute_kernel_config,
            # Seq shard axes, outermost (SP ring) first. TP×SP adds the TP axis so the score adds each
            # device's tp_rank*Sq' block-cyclic sub-offset — rotation-EXACT (vs the flat [] path, which is
            # linear-approximate under a mid-slab start). SP-only ([sp]) when the query stays SP-sharded.
            seq_shard_axes=[self.sp_axis, self.tp_axis] if tpsp else [self.sp_axis],
            cache_batch_idx=None,  # k_full is already sliced to this slot (batch-1) → no in-kernel select
            block_cyclic_sp_axis=self.sp_axis,
            block_cyclic_chunk_local=seq_len,  # cache slab == chunk_size_global / sp (== Sq'·tp when TP-split)
            kv_len=end_pos,
        )
        ttnn.deallocate(k_full)
        # wq_b replicated -> each chip already holds the COMPLETE head-summed logit, so there is NO
        # partial-logit all-reduce over tp. This is the win: the removed step was a 2-CCL (RS+AG) all-reduce
        # spanning the full end_pos-wide logit (+ a TILE<->ROW_MAJOR round-trip), the indexer's dominant cost.
        # Top-k key indices [1,1,S/sp,k] (ROW_MAJOR uint32). Future/pad -inf columns surface as the
        # 0xFFFFFFFF sentinel that sparse_mla drops. topk_large_indices: 16 <= k <= 2048, multiple of 16.
        # index_topk is a multiple of 16, so k is too iff end_pos is — assert it at the caller contract
        # (current chunk sizing guarantees tile alignment) rather than failing deep inside the op.
        assert end_pos % 16 == 0, f"indexer top-k requires a tile-aligned key count; got end_pos={end_pos}"
        # Block-cyclic logits are the full preallocated width T with a stale [end_pos, T) tail (kv_len only
        # wrote the real prefix); valid_length bounds top-k to that prefix so the tail is never read or ranked.
        topk_valid_length = end_pos
        idx = ttnn.experimental.topk_large_indices(
            logits, k=min(self.index_args.index_topk, end_pos), valid_length=topk_valid_length
        )
        # TP×SP: topk ran on the TP-seq-sharded rows ([1,1,S/(sp·tp),k]); regather over TP back to the
        # [1,1,S/sp,k] contract so sparse_sdpa/mla.py are unchanged. (Redundant TP-round-trip for GLM's
        # head→seq reshard, which re-splits it; correct regardless. tp=1: no-op.)
        if tpsp:
            # Regather the TP-seq-sharded top-k indices back to [1,1,S/sp,k]. topk_large_indices emits
            # ROW_MAJOR uint32, and an all-gather on a ROW_MAJOR tensor is routed by use_composite_all_gather
            # to composite_all_gather -> all_broadcast, whose multicast over a partial cluster-axis line of a
            # 2D (SP×TP) mesh DEADLOCKS the fabric (erisc routers stall in run_receiver_channel_step; device
            # unrecoverable, system_memory_manager.cpp TIMEOUT). Gather in TILE layout so it takes the NATIVE
            # minimal all-gather instead — the tile-aligned gather dim keeps it off the composite path, and
            # the native path handles this TP cluster-axis correctly (as _tp_rs_ag does, and as the canonical
            # top-k-index gather in tt_sampling.py does). Round-trip RM->TILE->gather->RM.
            idx_local = idx
            idx_tiled = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
            idx_gathered = self._tp_all_gather(idx_tiled, dim=2)  # native all-gather over TP; [1,1,S/sp,k] TILE
            idx = ttnn.to_layout(idx_gathered, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(idx_local)
            ttnn.deallocate(idx_tiled)
            ttnn.deallocate(idx_gathered)
        return idx


class NullIndexer:
    """Dense v3.1 stand-in for TtIndexer: forward() is a no-op returning None (no top-k, no K-cache
    write). Lets ttMLA bind self._indexer once at construction and call it unconditionally in forward.
    Mirrors TtIndexer.forward's contract — keep the two in sync if that signature/return changes."""

    def forward(self, *args, **kwargs):
        return None


class ReuseIndexer:
    """GLM-5.2 ``shared`` DSA layer stand-in: owns no indexer weights and never computes. The layer is
    still sparse (top-k SDPA) but reuses a prior ``full`` layer's top-k indices, injected at
    ttMLA.forward(indexer_indices=...). forward() is unreachable there (the injected indices short-
    circuit it); it raises if ever called, so a shared layer missing its reused indices fails loudly
    instead of silently going dense."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ReuseIndexer.forward called: a GLM-5.2 shared DSA layer must receive reused top-k indices "
            "via MLA.forward(indexer_indices=...)."
        )


# Back-compat alias; TtIndexer.WEIGHT_NAMES is the single source of truth.
INDEXER_WEIGHT_NAMES = TtIndexer.WEIGHT_NAMES


def resolve_has_indexer(config, state_dict=None, explicit=None, weight_cache_path=None, cache_name_prefix=None) -> bool:
    """Single source of truth for "is this a sparse DSA layer?", used by every ttMLA cache/check/
    build/load path so they cannot disagree. Resolution order:
      1. explicit override when not None,
      2. config.has_indexer when present (absence = unknown, NOT False),
      3. TtIndexer.matches_config(config) — runtime config carries DSA index_* fields,
      4. TtIndexer.has_host_weights(state_dict) — live from-weights callers,
      5. TtIndexer.check_cache_complete(...) — cache-only callers with a complete indexer cache,
      6. otherwise dense.
    Never resolve sparse detection through getattr(config, "has_indexer", False): a default-False
    flag silently disables sparse for cache-only construction (the bug this whole path fixes)."""
    if explicit is not None:
        return explicit
    flag = getattr(config, "has_indexer", None)
    if flag is not None:
        return bool(flag)
    if TtIndexer.matches_config(config):
        return True
    if TtIndexer.has_host_weights(state_dict):
        return True
    if weight_cache_path is not None and cache_name_prefix is not None:
        return TtIndexer.check_cache_complete(weight_cache_path, cache_name_prefix)
    return False


def indexer_layer_is_reused(config, layer_idx: int) -> bool:
    """GLM-5.2 ``shared`` layer: sparse attention but owns NO indexer (it reuses a prior ``full`` layer's
    top-k). True iff ``config.indexer_types[layer_idx] == "shared"``. Absent the map (v3.1 / v3.2 /
    GLM-5.1) every layer is a full indexer owner -> current behavior. Single source of truth for the
    device construction (ReuseIndexer binding) and the cache build (skip the indexer tensorbins)."""
    types = getattr(config, "indexer_types", None)
    return bool(types) and layer_idx < len(types) and types[layer_idx] == "shared"


def num_full_indexer_layers(config):
    """Count how many entries equal ``"full"`` in ``config.indexer_types``. Returns ``None`` when the list
    is absent or empty."""
    types = getattr(config, "indexer_types", None)
    if not types:
        return None
    return sum(1 for t in types if t == "full")


def full_indexer_rank(config, layer_idx: int) -> int:
    """Prefix rank over ``config.indexer_types``: count how many entries equal ``"full"`` before position
    ``layer_idx`` (exclusive), renumbering the matching positions into dense 0-based ranks. Returns
    ``layer_idx`` unchanged when the list is absent or empty."""
    types = getattr(config, "indexer_types", None)
    if not types:
        return layer_idx
    return sum(1 for t in types[:layer_idx] if t == "full")
