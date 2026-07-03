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
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup, interleaved_perm_matrix

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
        return weight_name.split(".")[-1]  # "indexer.wq_b" -> "wq_b"

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
        index_n_heads = getattr(config, "index_n_heads", 64)
        index_head_dim = getattr(config, "index_head_dim", 128)
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

        def repl(t, short):  # k_norm runs on the reduced 128-wide key, so it is replicated across TP
            return ttnn.as_tensor(
                t.contiguous().to(torch.bfloat16),
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

        def repl_t(t, short):  # host [out, in] -> device [in, out] REPLICATED (full, every chip)
            return ttnn.as_tensor(
                t.T.contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=_cache_name(short),
            )

        # PROTOTYPE (SP-seq-sharded indexer): wq_b is REPLICATED with all H_idx heads on every chip, so the
        # indexer_score head-sum is complete on-chip (no TP logit all-reduce). The queries are seq-sharded
        # over SP×TP instead. wk / weights_proj still contract over hidden (TP-sharded) → upload transposed
        # and sharded on that contraction axis → partials reduced by _tp_rs_ag.
        result = {
            "wq_b": repl_t(wq_b, "wq_b_repl"),  # [q_lora_rank, H_idx*D_idx] replicated (all heads)
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
        is_chunked: bool = False,
    ):
        """Architecture constants are read from the HF config (DS defaults below; GLM sets
        index_rope_interleave / index_n_heads etc.). θ / YaRN / rope table length come from the
        same config — single source of truth. Device index-key cache is grown by concat per chunk.

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
        self.weight_cache_path = weight_cache_path
        self.layer_idx = layer_idx
        self.tt_ccl = tt_ccl
        self.ccl_num_links = ccl_num_links
        self.ccl_topology = ccl_topology
        self.index_args = SimpleNamespace(
            index_n_heads=getattr(config, "index_n_heads", 64),
            index_head_dim=getattr(config, "index_head_dim", 128),
            index_topk=getattr(config, "index_topk", 2048),
            index_rope_interleave=getattr(config, "index_rope_interleave", False),
        )
        self.seq_len = seq_len
        self.is_chunked = is_chunked
        # Block-cyclic key-cache path (chunked prefill): mirrors the MLA KVPE cache — a persistent,
        # per-user, block-cyclic ND-sharded key cache written by update_padded_kv_cache and scored by
        # indexer_score_dsa's block-cyclic reader. Both variants use it; the rope is always the interleaved
        # on-device INDEXED op (rotary_embedding_indexed). GLM is natively interleaved; DS-v3.2's half-split
        # rope is reconciled by _rope_perm below (permute q/k rope halves so the interleaved op matches the
        # DS reference). Single-shot (all variants) stays on the natural gather+concat path.
        self._blockcyclic = is_chunked
        # Block-cyclic key cache (persistent [num_users,1,S/sp,D_idx]) is NOT owned here: exactly like the
        # MLA KVPE cache, the caller allocates it and passes it into forward(index_kv_cache=...) every call;
        # the indexer never self-allocates it. write_k typecasts the roped key to the cache's dtype before
        # the in-place write, so the caller controls the dtype (BF8 is validated — rotated + accuracy suites
        # match BF16 within bf16 noise, ~5e-4 PCC — so it can be allocated BF8 to halve the memory).
        # self._index_kbuf below is the NATURAL (single-shot) path's internal concat-grown cache only.
        self._index_kbuf = None
        self._upload_weights(idx_host)
        self._build_rope_tables()
        # DS block-cyclic uses the interleaved rotary_embedding_indexed op, but DS weights emit the
        # half-split (rotate_half) rope arrangement. Permute the rope half (half-split -> interleaved) so
        # the interleaved op pairs the right dims with each frequency; applied to BOTH q and k, the
        # permutation cancels in q·k, so the score (hence top-k) matches the DS half-split reference. GLM
        # is natively interleaved -> no permute. NOTE: the stored key is then in interleaved layout —
        # reindex by rope.interleaved_to_halfsplit_perm to compare it against a half-split reference.
        self._rope_perm = None
        if self._blockcyclic and not self.index_args.index_rope_interleave:
            # rope.interleaved_perm_matrix owns the half-split -> interleaved convention (single source).
            self._rope_perm = ttnn.from_torch(
                interleaved_perm_matrix(64).to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        # Block-cyclic QUERY rope tables (interleaved, natural order, replicated full-length). The SP×TP
        # seq-sharded query cannot use the on-device INDEXED rope op (its cluster_axis is single-axis and
        # can't express a 2-D SP×TP query shard), so _bc_query_rope ropes it host-side via
        # rotary_embedding_llama, which needs interleaved cos/sin + trans. Same interleaved base as the
        # block-cyclic KEY rope_tensors (get_cos_sin_matrix defaults to interleave=True), so q and k land in
        # the same basis. Built only on the block-cyclic path; the natural path uses self._idx_* instead.
        self._bc_q_cos = self._bc_q_sin = self._bc_q_trans = None
        if self._blockcyclic:
            q_tables = RotarySetup(self.config, self.mesh_device, sp_axis=self.sp_axis).get_indexer_rope_tables(
                interleave=True
            )
            self._bc_q_cos, self._bc_q_sin, self._bc_q_trans = q_tables["cos"], q_tables["sin"], q_tables["trans"]

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
        """All-gather across the TP axis → concatenated along ``dim``, replicated on TP. tp=1: no-op.
        Used to gather the SP×TP seq-sharded top-k indices back to the SP block's full row set."""
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

    def _build_rope_tables(self):
        """Precompute the indexer's natural-path RoPE tables via RotarySetup.get_indexer_rope_tables
        (single source of rope-convention logic). ``index_rope_interleave`` picks the layout + matching
        device op in ``_device_rope_pe``: DS (False) -> rotate_half, no trans_mat -> rotary_embedding_hf;
        GLM (True) -> interleaved + trans_mat -> rotary_embedding_llama (matches the MLA's own rope).
        Full-length replicated pure rotations (no mscale); ``_device_rope_pe`` slices per chunk. (The
        block-cyclic path instead reuses ttMLA's block-cyclic rope_tensors passed into forward.)"""
        tables = RotarySetup(self.config, self.mesh_device, sp_axis=self.sp_axis).get_indexer_rope_tables(
            interleave=self.index_args.index_rope_interleave
        )
        self._idx_cos, self._idx_sin, self._idx_trans = tables["cos"], tables["sin"], tables["trans"]

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

    def _device_rope_pe(self, x: ttnn.Tensor, glob: int, start_pos: int, shard_axes: tuple = ()) -> ttnn.Tensor:
        """On-device RoPE on the rope half (first 64) of the last dim.
        x [1, n_heads, S, D_idx]; cos/sin sliced to this chunk's global positions [start_pos,
        start_pos+glob), then mesh-partitioned (dim 2) over each mesh axis in ``shard_axes`` IN ORDER so
        each chip keeps the cos/sin for its own contiguous query block. Examples (positions of chip r's
        first row): () = unsharded full glob (K cache); (sp_axis,) = start_pos + sp*S_local; (sp_axis,
        tp_axis) = start_pos + sp*(tp_factor*S_local) + tp*S_local — the SP×TP seq-sharded indexer queries."""
        h, n = x.shape[1], x.shape[2]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, n, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, n, self.index_args.index_head_dim])
        cos = ttnn.slice(self._idx_cos, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        sin = ttnn.slice(self._idx_sin, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        for ax in shard_axes:  # successive partitions land each chip on its own contiguous position block
            if list(self.mesh_device.shape)[ax] > 1:
                cos = ttnn.mesh_partition(cos, dim=2, cluster_axis=ax)
                sin = ttnn.mesh_partition(sin, dim=2, cluster_axis=ax)
        if self._idx_trans is not None:  # GLM: interleaved RoPE (Meta-style + trans_mat)
            pe = ttnn.experimental.rotary_embedding_llama(pe, cos, sin, self._idx_trans, is_decode_mode=False)
        else:  # DS: non-interleaved (rotate_half)
            pe = ttnn.experimental.rotary_embedding_hf(
                pe, cos, sin, is_decode_mode=False, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )
        return ttnn.concat([pe, nope], dim=-1)

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

    def _bc_query_rope(self, x: ttnn.Tensor, glob: int, start_pos: int) -> ttnn.Tensor:
        """Block-cyclic QUERY rope for the SP×TP seq-sharded query (block-cyclic path only). The query is
        ONE contiguous chunk at global positions [start_pos, start_pos+glob), so — unlike the KEY rope
        (_bc_rope_pe, the on-device INDEXED op over the block-cyclic-stored cache) — it is roped host-side:
        slice the interleaved cos/sin to this chunk and mesh-partition over SP then TP so each chip keeps
        its own contiguous sq-row block (positions start_pos + sp*(tp*sq) + tp*sq). rotary_embedding_llama
        shares its rotation kernel with rotary_embedding_indexed, so — given the same interleaved cos/sin +
        trans for the same positions — the query lands in the SAME basis as the block-cyclic keys. DS:
        _rope_perm reorders the rope half (half-split -> interleaved) to match _bc_rope_pe (the perm cancels
        in q·k, applied to both q and k). GLM (natively interleaved) has _rope_perm=None -> no perm."""
        h, n = x.shape[1], x.shape[2]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, n, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, n, self.index_args.index_head_dim])
        if self._rope_perm is not None:  # DS: half-split -> interleaved (matches _bc_rope_pe's key rope)
            pe_i = ttnn.linear(pe, self._rope_perm, compute_kernel_config=self.hifi4_fp32_compute_kernel_config)
            ttnn.deallocate(pe)
            pe = pe_i
        cos = ttnn.slice(self._bc_q_cos, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        sin = ttnn.slice(self._bc_q_sin, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        for ax in (self.sp_axis, self.tp_axis):  # successive partitions -> each chip's own sq-row block
            if list(self.mesh_device.shape)[ax] > 1:
                cos = ttnn.mesh_partition(cos, dim=2, cluster_axis=ax)
                sin = ttnn.mesh_partition(sin, dim=2, cluster_axis=ax)
        pe = ttnn.experimental.rotary_embedding_llama(pe, cos, sin, self._bc_q_trans, is_decode_mode=False)
        out = ttnn.concat([pe, nope], dim=-1)
        ttnn.deallocate(pe)
        ttnn.deallocate(nope)
        return out

    def _gather_index_kbuf(self, index_kbuf: ttnn.Tensor) -> ttnn.Tensor:
        """Read the block-cyclic ND-sharded key cache back to a replicated full-T [num_users,1,T,D_idx]
        (block-cyclic order preserved, bf16 TILE) for indexer_score_dsa's block-cyclic reader — the
        analogue of ttMLA._gather_kvpe_prefix, but the score op consumes TILE so no RM/typecast step.
        The op selects this user's slot via cache_batch_idx; the unwritten suffix is never scored
        (future positions are causally masked).

        PERF TODO: this SP all-gather is currently a blocking barrier — it materializes the whole full-T
        key cache before indexer_score_dsa runs. It should instead be FUSED INTO the score op (ring-joint
        style, like ring_mla / ring-joint SDPA fuse the KV all-gather with the attention compute): pipeline
        the per-slab gather with the score matmul so each SP key slab is gathered and scored as it arrives,
        overlapping the CCL with the op's own compute instead of paying a full gather up front. Op-level
        change (ring indexer_score), not a host-side reorder."""
        cache_i = ttnn.to_memory_config(index_kbuf, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        full = self._sp_all_gather(cache_i, dim=2)  # [B,1,T,D_idx] replicated, block-cyclic
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        return full

    def write_k(self, hidden_states, seq_len, start_pos, rope_tensors=None, cache_user_id=0, index_kbuf=None):
        """Device K stem (wk + TP all-reduce + k_norm + device rope) written into the device index-key
        cache. forward() calls this on every chunk so the key-cache stays complete — else later chunks
        score against missing keys for the early prefix. (Dense v3.1 binds a NullIndexer, so write_k never
        runs there.) Two paths, fixed by self._blockcyclic:
          - block-cyclic (GLM/DS chunked): rope the PER-CHIP shard at its block-cyclic positions, then write
            it in place via update_padded_kv_cache (per-user slot, pad-aware kv_actual_global offset) — no
            SP all-gather, no O(n^2) concat; the cache stays SP-sharded.
          - natural (single-shot, all variants): SP all-gather to full-glob + natural rope + concat-grow."""
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

        if self._blockcyclic:
            # Rope the per-chip shard at its block-cyclic positions (kv_actual_global=start_pos), then
            # write it into this user's slot. update_padded_kv_cache places each chip's rows at the
            # block-cyclic offset (pad-aware) — the same math the query/key rope above uses.
            k = self._bc_rope_pe(k, rope_tensors, start_pos)  # [1, 1, S/sp, D_idx] bf16
            if k.dtype != index_kbuf.dtype:  # write dtype must match the cache (update_padded_kv_cache asserts)
                k = ttnn.typecast(k, index_kbuf.dtype)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                index_kbuf,
                k,
                slot_idx=cache_user_id,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=start_pos,
                cluster_axis=self.sp_axis,
            )
            ttnn.deallocate(k)
            return

        # Natural path (single-shot, all variants): SP all-gather to a full/replicated cache. write_k runs
        # once here (start_pos==0), so the concat-grow below is dead for current configs -- it only fired in
        # the retired non-block-cyclic chunked mode. The live cost is the all-gather + full replication
        # (glob keys/chip vs glob/sp block-cyclic).
        #
        # TODO(refactor, drop natural path): fold single-shot onto the block-cyclic
        # path (treat it as one full-seq chunk at start_pos=0): SP-sharded cache + in-place
        # update_padded_kv_cache, no all-gather / no replication / no concat -- then delete this branch,
        # _device_rope_pe's natural use, and _sp_all_gather. Dependency: single-shot must supply the
        # block-cyclic INDEXED rope tables (mla.py builds natural-order tables via _apply_rope_one_shot in
        # single-shot; _bc_rope_pe needs get_rope_tensors_indexed). Verify block-cyclic alignment holds for
        # single-shot seq lengths and that test_sparse_mla_vs_trace.py (is_chunked=False) passes unified.
        glob = seq_len * self.sp_factor
        k = self._sp_all_gather(k, dim=2)  # [1, 1, glob, D_idx] full, replicated, natural order
        k = self._device_rope_pe(k, glob, start_pos)  # on-device rope
        # The rest of this MLA code manually deallocates intermediate device tensors, so free the device
        # buffers we drop here too rather than relying on Python ref-loss (which would accumulate device
        # allocations over a long chunked prefill or across repeated requests).
        if start_pos == 0 or self._index_kbuf is None:
            if self._index_kbuf is not None:  # start_pos==0 reset: drop the prior request's full cache
                ttnn.deallocate(self._index_kbuf)
            self._index_kbuf = k
        else:
            old = self._index_kbuf
            self._index_kbuf = ttnn.concat([old, k], dim=2)  # concat copies into a fresh buffer...
            ttnn.deallocate(old)  # ...so the old cache and this chunk's keys are both free to drop
            ttnn.deallocate(k)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        qr: ttnn.Tensor,
        seq_len: int,
        start_pos: int = 0,
        rope_tensors: dict = None,
        cache_user_id: int = 0,
        index_kv_cache: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Indexer forward → top-k key indices [1, 1, S/sp, k] over the device index-key cache, SP-sharded
        on the query axis (each chip scores its own S/sp rows; no Q/W all-gather). Fully on-device:
        stems, RoPE, cache, logits, topk — no host.

        ``index_kv_cache`` (block-cyclic path): the persistent per-user key cache, allocated by the caller and
        passed in every call — the same ownership as ttMLA's KVPE ``kvpe_cache``. Required for the block-cyclic
        path (the indexer never self-allocates it); ignored by the natural (single-shot) path.

        ``qr`` is the shared q_a latent (q_a_proj + TP all-reduce + q_a_layernorm) — ttMLA computes it once
        and passes it in; the indexer applies wq_b to it (no q_a stem of its own). ``qr`` is NOT deallocated
        here — ttMLA's _q_stem consumes it afterwards. ``hidden_states`` is still needed for the K stem
        (write_k) and the per-head weights (weights_proj). (write_k is called internally here; ttMLA.forward
        only ever calls self._indexer.forward — it never calls write_k directly.)

        Block-cyclic path (GLM/DS chunked): ``rope_tensors`` (the MLA's block-cyclic indexed cos/sin/trans)
        and ``cache_user_id`` (per-user slot) drive the per-user block-cyclic key cache + block-cyclic
        scoring. Scoring currently spans the full preallocated cache width (see the kv_len TODO below).
        Natural path ignores rope_tensors/cache_user_id."""
        a = self.index_args
        glob = seq_len * self.sp_factor  # global query/key count this chunk
        end_pos = start_pos + glob
        # Block-cyclic key cache is caller-owned (like the KVPE cache) — required, never self-allocated.
        if self._blockcyclic:
            assert index_kv_cache is not None, (
                "block-cyclic indexer requires an externally-allocated index_kv_cache passed to forward() "
                "(same ownership as the MLA KVPE cache); none was provided"
            )
        self.write_k(
            hidden_states,
            seq_len,
            start_pos,
            rope_tensors=rope_tensors,
            cache_user_id=cache_user_id,
            index_kbuf=index_kv_cache,
        )

        # PROTOTYPE — SP×TP seq-sharded indexer. The expensive part of the old TP-head-sharded indexer was
        # all-reducing the full [S/sp, end_pos] partial logits over TP (two CCLs over a 50k-wide tensor).
        # Here every chip keeps ALL H_idx heads (wq_b replicated) so its head-sum is COMPLETE on-chip — no
        # logit all-reduce. To keep per-chip compute equal we shard the query sequence over TP as well: the
        # S/sp = seq_len rows are split into sq = seq_len/tp rows/chip, so each chip does H_idx × sq score
        # work (= the old (H_idx/tp) × seq_len). Only the small top-k indices are gathered over TP afterward.
        assert seq_len % self.tp_factor == 0, f"seq_len {seq_len} must be divisible by tp {self.tp_factor}"
        sq = seq_len // self.tp_factor  # this chip's query rows after the extra TP seq-shard
        assert sq % ttnn.TILE_SIZE == 0, f"per-chip indexer rows {sq} must be tile-aligned"

        # Q stem: shared q_a latent (qr, SP-sharded seq) -> shard seq over TP too -> all-heads wq_b.
        qr_sp = ttnn.mesh_partition(qr, dim=2, cluster_axis=self.tp_axis)  # [1,1,sq,q_lora] this chip's rows
        q = ttnn.linear(
            qr_sp,
            self._idx_wq_b,  # replicated: [q_lora, H_idx*D_idx]
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,sq, H_idx*D_idx]
        ttnn.deallocate(qr_sp)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=a.index_n_heads, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, H_idx, sq, D_idx] — all heads resident (wq_b replicated, no TP head-split)
        # SP×TP query rope at natural positions start_pos + sp*(tp*sq) + tp*sq (cos/sin sharded over SP then
        # TP). The block-cyclic keys are roped in the interleaved INDEXED basis, but that op's cluster_axis
        # is single-axis and can't express a 2-D SP×TP query seq-shard — so match its basis host-side via
        # _bc_query_rope (perm + rotary_embedding_llama, which shares the indexed op's rotation kernel).
        # Natural (single-shot) keys are rotate_half, so the query uses the same natural _device_rope_pe.
        if self._blockcyclic:
            q_dev = self._bc_query_rope(q, glob, start_pos)
        else:
            q_dev = self._device_rope_pe(q, glob, start_pos, shard_axes=(self.sp_axis, self.tp_axis))

        # weights_proj: full all-reduce over TP (all H_idx heads, not the rs-only head split), scale, then
        # shard seq over TP to match q -> [1, H_idx, sq, 1].
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wts = self._tp_rs_ag(wts)  # all-reduce (RS+AG) over tp -> full [1,1,seq_len,H_idx], replicated on tp
        wts = ttnn.multiply(wts, a.index_n_heads**-0.5 * a.index_head_dim**-0.5)
        wts = ttnn.mesh_partition(wts, dim=2, cluster_axis=self.tp_axis)  # -> [1,1,sq,H_idx] this chip's rows
        weights = ttnn.permute(wts, (0, 3, 2, 1))  # [1, H_idx, sq, 1]
        ttnn.deallocate(wts)

        # Causality is fused inside indexer_score_dsa (future columns -> -inf from chunk_start_idx). All
        # H_idx heads are resident on-chip (wq_b replicated), so each chip's logit is COMPLETE — no TP
        # all-reduce of partial logits. cluster_axis=None -> the op derives this chip's causal offset from
        # its FLATTENED row-major mesh index r = sp*tp_factor + tp, so chunk_start = start_pos + r*sq =
        # start_pos + sp*(tp_factor*sq) + tp*sq — exactly the chip's global first query row. The op REQUIRES
        # cluster_axis=None for a seq shard across BOTH axes (block_cyclic_chunk_local == sq*tp, tp>1): a
        # named axis would miss the second axis's offset (it TT_FATALs on that combination).
        # Kernel tiling: head_group_size=0 keeps ALL H_idx heads resident -> the key cache is read ONCE (vs
        # Hi/HB times when streaming), the dominant cost here. That needs L1 headroom, so k_chunk shrinks to
        # 64. q_chunk (<=64) must divide sq/TILE; 32 is the fallback when 64 doesn't (e.g. sq=160).
        q_chunk = next(c for c in (64, 32) if (sq // ttnn.TILE_SIZE) % (c // ttnn.TILE_SIZE) == 0)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=min(64, end_pos), head_group_size=0)
        if self._blockcyclic:
            # Gather the per-user block-cyclic key cache to replicated full-T; the op reads it back in
            # logical order via invP (keyed on block_cyclic_sp_axis / _chunk_local, independent of
            # cluster_axis), selects this user's slot via cache_batch_idx, and straddles a non-slab-aligned
            # start_pos. Scores the FULL preallocated width T: positions past each query are causally -inf,
            # and top-k below drops those (-inf -> sentinel).
            k_full = self._gather_index_kbuf(index_kv_cache)  # [num_users, 1, T, D_idx] bf16 TILE, block-cyclic
            logits = ttnn.experimental.indexer_score_dsa(
                q_dev,
                k_full,
                weights,
                chunk_start_idx=start_pos,
                program_config=cfg,
                cluster_axis=None,  # flattened SP×TP causal offset (required for the both-axes query shard)
                cache_batch_idx=cache_user_id,
                block_cyclic_sp_axis=self.sp_axis,
                block_cyclic_chunk_local=seq_len,  # per-SP-chip cache slab == chunk_size_global / sp
            )
            ttnn.deallocate(k_full)
        else:
            logits = ttnn.experimental.indexer_score_dsa(
                q_dev,
                self._index_kbuf,
                weights,
                chunk_start_idx=start_pos,
                program_config=cfg,
                cluster_axis=None,  # flattened SP×TP causal offset
            )  # [1, 1, sq, end_pos] ROW_MAJOR, fully head-summed
        ttnn.deallocate(q_dev)
        ttnn.deallocate(weights)

        # Top-k over this chip's sq rows -> [1,1,sq,k]. Then gather the TP seq-shards back to the SP block's
        # full seq_len rows (replicated across TP) so the indices match sparse_mla's SP-sharded q.
        assert end_pos % 16 == 0, f"indexer top-k requires a tile-aligned key count; got end_pos={end_pos}"
        idx_local = ttnn.experimental.topk_large_indices(logits, k=min(self.index_args.index_topk, end_pos))
        ttnn.deallocate(logits)
        return self._tp_all_gather(idx_local, dim=2)  # [1,1,seq_len,k] replicated on TP


class NullIndexer:
    """Dense v3.1 stand-in for TtIndexer: forward() is a no-op returning None (no top-k, no K-cache
    write). Lets ttMLA bind self._indexer once at construction and call it unconditionally in forward.
    Mirrors TtIndexer.forward's contract — keep the two in sync if that signature/return changes."""

    def forward(self, *args, **kwargs):
        return None


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
