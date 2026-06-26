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

from types import SimpleNamespace

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix, get_rot_transformation_mat

# DSA indexer weights (v3.2 / GLM). When present in the state_dict they are popped before the
# v3 weight loader (in ttMLA.__init__) and routed here; absent for dense v3.1.
INDEXER_WEIGHT_NAMES = ("indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.k_norm_bias", "indexer.weights_proj")


class TtIndexer:
    """DSA lightning indexer for one MLA layer. Self-contained: owns the indexer weights, the
    grown-by-concat device index-key cache and the indexer RoPE tables, and runs its own TP/SP
    collectives. All MLA-layer dependencies it reuses are injected at construction (no ttMLA ref)."""

    def __init__(
        self,
        idx_host,
        *,
        config,
        mesh_device,
        sp_axis: int,
        tp_axis: int,
        scale: float,
        default_compute_kernel_config,
        hifi4_fp32_compute_kernel_config,
        weight_cache_path,
        layer_idx: int,
        tt_ccl,
        ccl_num_links: int,
        ccl_topology,
    ):
        """Architecture constants are read from the HF config (DS defaults below; GLM sets
        index_rope_interleave / index_n_heads etc.). θ / YaRN / rope table length come from the
        same config — single source of truth. Device index-key cache is grown by concat per chunk.

        Injected from ttMLA (the indexer keeps no back-reference): the SP×TP mesh + axes,
        compute-kernel configs, softmax scale, weight-cache location, and the CCL handles used by the
        inlined collectives (_tp_rs_ag / _sp_all_gather). The q_a latent (qr) is passed into forward(),
        not held here — so the indexer holds no MLA weights."""
        self.config = config
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        mesh_shape = list(mesh_device.shape)
        self.sp_factor = mesh_shape[sp_axis]
        self.tp_factor = mesh_shape[tp_axis]
        self.scale = scale
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
        self._index_kbuf = None
        self._upload_weights(idx_host)
        self._build_rope_tables()

    # Inlined TP/SP collectives — the indexer owns its own copy so it depends on tt_ccl, not on ttMLA
    # (the dense MLA forward keeps its own equivalents; both go through the same tt_ccl handles).
    def _tp_rs_ag(self, t, rs_only=False):
        """All-reduce over TP = reduce-scatter (dim 3) then all-gather; rs_only stops after the RS."""
        t = (
            ttnn.experimental.reduce_scatter_minimal_async(
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
            if self.tp_factor > 1
            else t
        )
        if rs_only or self.tp_factor == 1:
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
        """All-gather (concat) across the TP axis → replicated on TP. tp=1: no-op. Used by the
        head→sequence reshuffle so each chip regains all H_idx heads before indexer_score."""
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
        """Precompute device cos/sin for the indexer RoPE via the shared builder
        (``get_cos_sin_matrix``). ``index_rope_interleave`` picks the layout + matching device op:
        DS (False) -> rotate_half (halves [c0,c1,..,c0,c1,..], no trans_mat) -> rotary_embedding_hf;
        GLM (True) -> interleaved (duplicated pairs [c0,c0,c1,c1,..] + trans_mat) ->
        rotary_embedding_llama (matches the MLA's own rope). Tables are pure (default
        ``bake_mscale=False`` — mscale lives in ``self.scale``). Op selection stays in ``_device_rope_pe``.

        Frequencies come from the HF ``self.config`` — the single source of truth and identical to the
        MLA's own rope (same θ / YaRN); the builder always applies YaRN, which is a no-op at
        ``rope_factor==1`` (GLM). Tables are full-length; ``_device_rope_pe`` slices per chunk."""
        interleave = self.index_args.index_rope_interleave
        cos, sin = get_cos_sin_matrix(self.config, interleave=interleave)  # bake_mscale=False → pure tables
        repl = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._idx_cos, self._idx_sin = repl(cos), repl(sin)
        self._idx_trans = repl(get_rot_transformation_mat()) if interleave else None

    def _upload_weights(self, w):
        """Indexer weights → device (stems on device, replicated across TP).

        Rides the same on-disk weight cache as the MLA weights: each tensor goes through
        ``ttnn.as_tensor`` with a ``layer_{idx}.mla.indexer_*`` cache file under the layer's
        weight_cache_path (same dir + scheme as `_convert_and_cache_weights`), so a second load
        reads the converted/sharded tensor from disk instead of re-converting on host.

        wk / weights_proj contract over `dim` (hidden is TP-sharded on dim), so they
        are uploaded transposed and sharded on that contraction axis → matmul yields
        per-chip partials reduced by _tp_rs_ag. wq_b is column-parallel (sharded on its
        H_idx*D_idx output) so each chip builds H_idx/tp indexer heads; qr is
        replicated so no reduce is needed. k_norm runs on the reduced 128-wide key (replicated).
        """

        def _cache_name(name):
            cp = self.weight_cache_path
            return str(cp / f"layer_{self.layer_idx}.mla.indexer_{name}") if cp else None

        def repl(t, name):
            return ttnn.as_tensor(
                t.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=_cache_name(name),
            )

        def shard(t, axis, name):  # t [out, in] -> device [in, out], tensor dim `axis` sharded across tp_axis
            dims = [None, None]
            dims[self.tp_axis] = axis
            return ttnn.as_tensor(
                t.T.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
                ),
                cache_file_name=_cache_name(name),
            )

        def shard_in(t, name):  # device [in, out] sharded on `in` (contraction axis) across tp_axis
            return shard(t, 0, name)

        # wq_b is column-parallel: shard its H_idx*D_idx OUTPUT across tp so
        # each chip builds only H_idx/tp indexer heads. qr is replicated, so no reduce is needed; the
        # per-head logits are summed across tp by an all-reduce after indexer_score (see topk).
        self._idx_wq_b = shard(w["indexer.wq_b"], 1, "wq_b")  # [q_lora_rank, H_idx*D_idx] col-sharded on out
        self._idx_wk = shard_in(w["indexer.wk"], "wk")  # [dim, D_idx] sharded on dim
        self._idx_wproj = shard_in(w["indexer.weights_proj"], "weights_proj")  # [dim, H_idx] sharded on dim
        self._idx_knorm_w = repl(w["indexer.k_norm"], "k_norm")  # [D_idx]
        self._idx_knorm_b = repl(w["indexer.k_norm_bias"], "k_norm_bias")  # [D_idx]

    def _device_rope_pe(self, x: ttnn.Tensor, glob: int, start_pos: int, sp_shard: bool = False) -> ttnn.Tensor:
        """On-device RoPE on the rope half (first 64) of the last dim.
        x [1, n_heads, S, D_idx]; cos/sin sliced to this chunk's global positions [start_pos,
        start_pos+glob). With sp_shard the cos/sin are mesh-partitioned across SP so each chip
        ropes its own contiguous query block (positions start_pos + sp_rank*S_local) — used for the
        SP-sharded indexer queries (the K cache stays full/gathered, so it ropes unsharded)."""
        h, n = x.shape[1], x.shape[2]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, n, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, n, self.index_args.index_head_dim])
        cos = ttnn.slice(self._idx_cos, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        sin = ttnn.slice(self._idx_sin, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        if sp_shard and self.sp_factor > 1:  # each SP chip keeps its block's cos/sin (natural-order shard)
            cos = ttnn.mesh_partition(cos, dim=2, cluster_axis=self.sp_axis)
            sin = ttnn.mesh_partition(sin, dim=2, cluster_axis=self.sp_axis)
        if self._idx_trans is not None:  # GLM: interleaved RoPE (Meta-style + trans_mat)
            pe = ttnn.experimental.rotary_embedding_llama(pe, cos, sin, self._idx_trans, is_decode_mode=False)
        else:  # DS: non-interleaved (rotate_half)
            pe = ttnn.experimental.rotary_embedding_hf(
                pe, cos, sin, is_decode_mode=False, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )
        return ttnn.concat([pe, nope], dim=-1)

    def _chunk_offset(self, start_pos: int, seq_len: int, row_split: bool = False) -> ttnn.Tensor:
        """Per-device causal chunk-start (in TILES) for indexer_score's fused causal mask. Each device's
        first query row sits at this absolute position; the op masks keys beyond (offset·32 + local_row).

        Default (head-parallel): each SP chip owns the contiguous query block [sp_rank*seq_len, ...), so its
        start is start_pos + sp_rank*seq_len (seq_len = S_local), replicated across TP.

        row_split (the head→sequence reshuffle): query rows are ALSO sharded over TP, so chip (sp=i, tp=r)
        owns the sub-block [i*seq_len + r*(seq_len/tp), ...) and its start must include the TP offset —
        otherwise tp_rank>0 rows get the wrong causal diagonal. Sharded over BOTH mesh axes."""
        sp = self.sp_factor
        if not row_split:
            vals = torch.zeros(sp, 1, 32, 32, dtype=torch.int32)
            for i in range(sp):
                vals[i, 0, 0, 0] = (start_pos + i * seq_len) // 32
            dims = [None, None]
            dims[self.sp_axis] = 0  # shard the sp blocks across the SP axis, replicate across TP
        else:
            tp = self.tp_factor
            sub = seq_len // tp  # rows per (sp, tp) chip
            vals = torch.zeros(sp, tp, 32, 32, dtype=torch.int32)
            for i in range(sp):
                for r in range(tp):
                    vals[i, r, 0, 0] = (start_pos + i * seq_len + r * sub) // 32
            dims = [None, None]
            dims[self.sp_axis] = 0  # sp sub-blocks across SP axis
            dims[self.tp_axis] = 1  # tp sub-blocks across TP axis
        return ttnn.from_torch(
            vals,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
        )

    def write_k(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int):
        """Device K stem (wk + TP all-reduce + k_norm + SP all-gather + device rope),
        appended to the device index-key cache. Runs on EVERY chunk — dense included —
        else later DSA chunks score against missing keys for the early prefix."""
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
        glob = seq_len * self.sp_factor
        k = self._sp_all_gather(k, dim=2)  # [1, 1, glob, D_idx] full, replicated, natural order
        k = self._device_rope_pe(k, glob, start_pos)  # on-device non-interleaved rope
        # Grow the replicated device cache by concat (natural order; no block-cyclic).
        # FOLLOW-UP (scoped redesign, not a drop-in): concat reallocates the whole key-cache each chunk
        # (O(n^2) copies over a long prefill). Reusing the MLA block-cyclic KVPE cache machinery
        # (update_padded_kv_cache + a gather/un-rotate read like _gather_kvpe_prefix) would avoid that,
        # but it changes the indexer key-cache layout (replicated-natural -> block-cyclic-SP) and the
        # scoring read path, so it is tracked as its own task rather than done here.
        self._index_kbuf = (
            k if start_pos == 0 or self._index_kbuf is None else ttnn.concat([self._index_kbuf, k], dim=2)
        )

    def forward(
        self, hidden_states: ttnn.Tensor, qr: ttnn.Tensor, seq_len: int, start_pos: int = 0, reshard: bool = False
    ) -> ttnn.Tensor:
        """Indexer forward → top-k key indices [1, 1, S/sp, k] over the device index-key cache, SP-sharded
        on the query axis (each chip scores its own S/sp rows; no Q/W all-gather). Fully on-device:
        stems, RoPE, cache, logits, topk — no host. K stays full/replicated (every query scores all keys).

        ``qr`` is the shared q_a latent (q_a_proj + TP all-reduce + q_a_layernorm) — ttMLA computes it once
        and passes it in; the indexer applies wq_b to it (no q_a stem of its own). ``qr`` is NOT deallocated
        here — ttMLA's _q_stem consumes it afterwards. ``hidden_states`` is still needed for the K stem
        (write_k) and the per-head weights (weights_proj). (write_k is also a public entry point: forward
        calls it, and ttMLA.forward calls it directly on dense chunks to keep the key-cache warm.)"""
        a = self.index_args
        glob = seq_len * self.sp_factor  # global query/key count this chunk
        end_pos = start_pos + glob
        self.write_k(hidden_states, seq_len, start_pos)

        # Q stem: the shared q_a latent (qr) -> indexer wq_b.
        q = ttnn.linear(
            qr,
            self._idx_wq_b,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, S/sp, (H_idx/tp)*D_idx] — queries stay SP-sharded (no all-gather; each chip scores its own rows)
        heads_local = a.index_n_heads // self.tp_factor  # this chip's indexer heads (col-parallel wq_b)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads_local, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, H_idx/tp, S/sp, D_idx]
        q_dev = self._device_rope_pe(q, glob, start_pos, sp_shard=True)  # per-chip query positions

        if reshard:
            # Transpose the TP shard heads→sequence so each chip holds ALL H_idx indexer heads for a
            # DISTINCT row slice. RoPE is already baked per-row (head-agnostic) so it rides along untouched.
            # indexer_score then needs no head all-reduce, and top-k is born sharded over sp·tp — exactly
            # sparse_sdpa's layout (see ttMLA._sparse_mla), so the two ops sit in one shared regime.
            q_dev = self._tp_all_gather(q_dev, dim=1)  # [1, H_idx, S/sp, D_idx] replicated on TP
            q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=self.tp_axis)  # [1, H_idx, S/(sp·tp), D_idx]

        # weights_proj: device stem. head-parallel → reduce-scatter so each chip keeps its H_idx/tp slice;
        # reshard → FULL all-reduce (every chip needs all H_idx weights) then split rows over TP.
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wts = self._tp_rs_ag(wts) if reshard else self._tp_rs_ag(wts, rs_only=True)
        wts = ttnn.multiply(wts, a.index_n_heads**-0.5 * self.scale)  # [1, 1, S/sp, H_idx (reshard) | H_idx/tp]
        if reshard:
            wts = ttnn.mesh_partition(wts, dim=2, cluster_axis=self.tp_axis)  # [1, 1, S/(sp·tp), H_idx]

        # Causality is fused inside indexer_score (future columns -> -inf from chunk_start_idx),
        # so no triu mask add here. Each chip scores only its H_idx/tp heads -> a PARTIAL logit
        # (the head-sum is separable; the -inf mask is head-independent so it survives the sum).
        # HB=0 keeps all H_idx/tp heads resident (fits L1 for tp>=2, i.e. <=32 heads/chip); tp=1 has
        # all 64 heads on one chip and must head-stream (HB=16).
        # heads resident per chip: full H_idx under reshard, else the H_idx/tp shard. >32 must head-stream.
        hb = (16 if a.index_n_heads > 32 else 0) if reshard else (0 if self.tp_factor > 1 else 16)
        # Indexer kernel knobs: QC=2 (q_chunk=64), KC=8 (k_chunk=256) capped to Skv/32 (the op requires
        # KC <= Skv/32; inert at the model's DSA K). HB=0 keeps all heads resident; HB=16 streams.
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=min(256, end_pos), head_group_size=hb)
        # Per-device causal offset: head-parallel → per-SP-rank start (start_pos + sp_rank*S_local); reshard →
        # per-(sp,tp) start including the TP row sub-offset. Either way each chip masks its own rows correctly.
        offset = self._chunk_offset(start_pos, seq_len, row_split=reshard)
        # indexer_score wants per-head weights [1, H_idx*, Sq, 1]; wts is [1, 1, Sq, H_idx*] (* = full or /tp).
        weights = ttnn.permute(wts, (0, 3, 2, 1))
        logits = ttnn.experimental.indexer_score(
            q_dev, self._index_kbuf, weights, chunk_start_idx=start_pos, program_config=cfg, chunk_offset=offset
        )
        # head-parallel: all-reduce(SUM) the partial logits over tp → full head-summed logit before top-k
        # (op emits ROW_MAJOR; _tp_rs_ag runs in TILE, so round-trip). reshard: heads are all local already,
        # logits are full → no reduction, and top-k is sharded over sp·tp to match the resharded attention q.
        if self.tp_factor > 1 and not reshard:
            # The op emits ROW_MAJOR; round-trip to TILE for the all-reduce. Passing RM straight to the
            # CCL is correct but ~10 ms slower — ttnn's RM reduce_scatter/all_gather tilize-with-padding
            # internally and add RM concats, costing more than this explicit ~6 ms tilize/untilize.
            logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT)
            logits = self._tp_rs_ag(logits)  # RS+AG over tp_axis == all-reduce SUM (reduce accumulates fp32)
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        # Top-k key indices [1,1,S/sp,k] (ROW_MAJOR uint32). Future/pad -inf columns surface as the
        # 0xFFFFFFFF sentinel that sparse_mla drops. topk_large_indices: 16 <= k <= 2048, multiple of 16.
        return ttnn.experimental.topk_large_indices(logits, k=min(self.index_args.index_topk, end_pos))
