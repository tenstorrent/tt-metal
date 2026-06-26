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
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix, get_rot_transformation_mat

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
        self._index_kbuf = None
        self._upload_weights(idx_host)
        self._build_rope_tables()

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
        """Precompute device cos/sin for the indexer RoPE via the shared builder
        (``get_cos_sin_matrix``). ``index_rope_interleave`` picks the layout + matching device op:
        DS (False) -> rotate_half (halves [c0,c1,..,c0,c1,..], no trans_mat) -> rotary_embedding_hf;
        GLM (True) -> interleaved (duplicated pairs [c0,c0,c1,c1,..] + trans_mat) ->
        rotary_embedding_llama (matches the MLA's own rope). Tables are pure rotations (no mscale
        baked in, matching the reference indexer). Op selection stays in ``_device_rope_pe``.

        Frequencies come from the HF ``self.config`` — the single source of truth and identical to the
        MLA's own rope (same θ / YaRN); the builder always applies YaRN, which is a no-op at
        ``rope_factor==1`` (GLM). Tables are full-length; ``_device_rope_pe`` slices per chunk."""
        interleave = self.index_args.index_rope_interleave
        cos, sin = get_cos_sin_matrix(self.config, interleave=interleave)  # pure rotation tables (no mscale)
        repl = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._idx_cos, self._idx_sin = repl(cos), repl(sin)
        self._idx_trans = repl(get_rot_transformation_mat()) if interleave else None

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

    def write_k(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int):
        """Device K stem (wk + TP all-reduce + k_norm + SP all-gather + device rope),
        appended to the device index-key cache. forward() calls this on every chunk so the key-cache
        stays complete — else later chunks score against missing keys for the early prefix. (Dense v3.1
        binds a NullIndexer instead, so write_k never runs there.)"""
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

    def forward(self, hidden_states: ttnn.Tensor, qr: ttnn.Tensor, seq_len: int, start_pos: int = 0) -> ttnn.Tensor:
        """Indexer forward → top-k key indices [1, 1, S/sp, k] over the device index-key cache, SP-sharded
        on the query axis (each chip scores its own S/sp rows; no Q/W all-gather). Fully on-device:
        stems, RoPE, cache, logits, topk — no host. K stays full/replicated (every query scores all keys).

        ``qr`` is the shared q_a latent (q_a_proj + TP all-reduce + q_a_layernorm) — ttMLA computes it once
        and passes it in; the indexer applies wq_b to it (no q_a stem of its own). ``qr`` is NOT deallocated
        here — ttMLA's _q_stem consumes it afterwards. ``hidden_states`` is still needed for the K stem
        (write_k) and the per-head weights (weights_proj). (write_k is called internally here; ttMLA.forward
        only ever calls self._indexer.forward — it never calls write_k directly.)"""
        a = self.index_args
        glob = seq_len * self.sp_factor  # global query/key count this chunk
        end_pos = start_pos + glob
        self.write_k(hidden_states, seq_len, start_pos)

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
        )  # [1, H_idx, sq, D_idx] — all heads resident
        # Per-chip positions are start_pos + sp*(tp*sq) + tp*sq: shard cos/sin over SP then TP (natural order).
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

        # Causality fused in indexer_score_dsa. cluster_axis=None -> the op derives this chip's offset from
        # its FLATTENED row-major mesh index r = sp*tp_factor + tp (device-coord order), so chunk_start =
        # start_pos + r*sq = start_pos + sp*(tp_factor*sq) + tp*sq — exactly the chip's global first row.
        # All H_idx heads are resident, so the logit is COMPLETE (no TP all-reduce).
        # Kernel tiling: head_group_size=0 keeps ALL H_idx heads resident -> the 50k-key cache is read ONCE
        # (vs Hi/HB times when streaming) — the dominant cost here. That needs L1 headroom, so k_chunk is
        # shrunk to 64 (the largest that fits 64 resident heads; 128 overflows). q_chunk capped at 64 and
        # must divide sq (Sqt % QC == 0); 32 is the fallback when 64 doesn't (e.g. sq=160).
        q_chunk = next(c for c in (64, 32) if (sq // ttnn.TILE_SIZE) % (c // ttnn.TILE_SIZE) == 0)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=min(64, end_pos), head_group_size=0)
        logits = ttnn.experimental.indexer_score_dsa(
            q_dev,
            self._index_kbuf,
            weights,
            chunk_start_idx=start_pos,
            program_config=cfg,
            cluster_axis=None,  # flattened SP×TP index -> correct 2D causal offset
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
