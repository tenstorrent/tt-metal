# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3.2 MLA: v3 MLA + DeepSeek Sparse Attention (DSA).

Subclasses the v3 ttMLA; projections, RoPE, cache and the e2e plumbing stay
v3. The DSA path activates only when needed (seq > index_topk; below it dense
== sparse by construction, so super().forward is used unchanged).

DSA forward = v3 forward with ring SDPA swapped for ops.sparse_mla (indices
replace the mask). v3's single-shot path materializes V before SDPA; we keep
attention latent and apply wkv_b2 after, like v3's chunked path.

Functional bring-up shortcuts (documented in status.md):
  - indexer stems (wq_b/wk/k_norm/weights_proj) on device, replicated across TP;
    only the non-interleaved RoPE stays on host (F1) — pe slices read back per chunk;
  - sparse attention via ops.sparse_mla CPU fallback;
  - sp=1 only (indexer needs full local seq; 2x2 later).
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA as _ttMLAv3
from models.demos.deepseek_v32.reference_cpu.model import ModelArgs
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.tt import ops

INDEXER_WEIGHT_NAMES = ("indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.k_norm_bias", "indexer.weights_proj")


def interleaved_to_halfsplit_perm(rope_dim: int = 64) -> torch.Tensor:
    """Dim permutation that maps this MLA's RoPE layout to vLLM's (and back — it is
    its own structural counterpart applied to the other side).

    RoPE CONVENTION NOTE (verified vs DeepSeek-V3.2-Exp reference, layers 0/30/60).
    This MLA carries q_pe and stores k_pe in the **interleaved** layout of the
    official ``inference/model.py`` (``apply_rotary_emb(interleaved=True)``: the
    2-D rotated pairs are dims (0,1),(2,3),...). vLLM's ``DeepseekV32`` uses the HF
    **rotate_half / half-split** layout (pairs (i, i+rope_dim/2)) with projection
    weights pre-permuted to suit. The two are exactly one fixed dim permutation
    apart, so:
      * q·k — and therefore the entire MLA output — is IDENTICAL in both layouts
        (the dot product sums over the rope dims, which the permutation only
        reorders). Measured: output PCC unchanged at 0.99983 either way.
      * the stored k_pe *values* differ element-wise: a direct comparison of our
        k_pe against a vLLM-written cache reads ~0.43 PCC, while the SAME tensor
        reindexed by this permutation matches at 0.99997.

    So within our self-consistent stack the layout is irrelevant. It matters ONLY
    when interoperating with a vLLM-written KV cache (e.g. cross-stack disaggregated
    prefill/decode, or validating against vLLM's recorded k_pe): reindex the rope
    half of the cache row with ``kpe[..., interleaved_to_halfsplit_perm()]`` to put
    it in vLLM's layout (or the reverse to ingest a vLLM cache).

    Returns the index tensor p such that ``interleaved_kpe[..., p] == halfsplit_kpe``:
    p = [0, 2, 4, ..., 1, 3, 5, ...] (even dims = cos halves, then odd dims = sin).
    """
    return torch.cat([torch.arange(0, rope_dim, 2), torch.arange(1, rope_dim, 2)])


class ttMLA(_ttMLAv3):
    """V3.2 MLA with DSA. Dense passthrough of v3 at seq <= index_topk."""

    def __init__(self, config, weights, mesh_device, **kwargs):
        # Indexer weights are v32-only — pop before v3 sees them, upload to device
        # after super().__init__ (which sets mesh_device / tp_axis / tp_factor).
        idx_host = {n: weights.pop(f"{n}.weight") for n in INDEXER_WEIGHT_NAMES if f"{n}.weight" in weights}
        # ModelArgs supplies the indexer's architecture constants (heads/dims/θ/YaRN factors); the rope
        # TABLE LENGTH comes from the HF config — the single source of truth — so it scales with the
        # model (production 128k). YaRN itself is applied unconditionally in _build_index_rope_tables
        # (matching the HF MLA rope, which is always-on), so the table length is decoupled from the
        # long-context gate and config.max_seq_len can be sized straight to the sequence.
        self.index_args = ModelArgs(max_batch_size=1, max_seq_len=int(config.max_seq_len))
        super().__init__(config, weights, mesh_device, **kwargs)
        self._has_indexer = bool(idx_host)
        # Device index key cache (backlog 19): replicated, natural order, grown by
        # concat per chunk. None until the first chunk; reset at start_pos==0.
        self._index_kbuf = None
        if self._has_indexer:
            self._upload_indexer_weights(idx_host)
            self._build_index_rope_tables()

    def _build_index_rope_tables(self):
        """Precompute device cos/sin for the indexer's non-interleaved RoPE
        (backlog 19 / issue #4). HF rotate_half layout: halves repeated → [1,1,max_seq,64].
        Probe-verified vs reference apply_rotary_emb(interleaved=False), PCC 0.99999."""
        # YaRN forced on (apply_yarn=True): the indexer is a long-context rope like the MLA path, so the
        # table length (config.max_seq_len) is independent of the long-context gate. Per-position freqs
        # are identical to the reference's (gated-on at 16384) — they depend only on original_seq_len.
        freqs = precompute_freqs_cis(self.index_args, apply_yarn=True)  # [max_seq, 32] complex; rope half = 64
        cos = torch.cat([freqs.real, freqs.real], dim=-1).reshape(1, 1, freqs.shape[0], 64).to(torch.bfloat16)
        sin = torch.cat([freqs.imag, freqs.imag], dim=-1).reshape(1, 1, freqs.shape[0], 64).to(torch.bfloat16)
        repl = lambda t: ttnn.from_torch(
            t,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._idx_cos, self._idx_sin = repl(cos), repl(sin)

    def _upload_indexer_weights(self, w):
        """Indexer weights → device (backlog 6: stems on device, replicated across TP).

        wk / weights_proj contract over `dim` (hidden is TP-sharded on dim), so they
        are uploaded transposed and sharded on that contraction axis → matmul yields
        per-chip partials reduced by _tp_rs_ag. wq_b is column-parallel (sharded on its
        H_idx*D_idx output) so each chip builds H_idx/tp indexer heads (change B); qr is
        replicated so no reduce is needed. k_norm runs on the reduced 128-wide key (replicated).
        """

        def repl(t):
            return ttnn.from_torch(
                t.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        def shard(t, axis):  # t [out, in] -> device [in, out], tensor dim `axis` sharded across tp_axis
            dims = [None, None]
            dims[self.tp_axis] = axis
            return ttnn.from_torch(
                t.T.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
                ),
            )

        def shard_in(t):  # device [in, out] sharded on `in` (contraction axis) across tp_axis
            return shard(t, 0)

        # wq_b is column-parallel (backlog 21 / change B): shard its H_idx*D_idx OUTPUT across tp so
        # each chip builds only H_idx/tp indexer heads. qr is replicated, so no reduce is needed; the
        # per-head logits are summed across tp by an all-reduce after indexer_score (see _indexer_topk).
        self._idx_wq_b = shard(w["indexer.wq_b"], axis=1)  # [q_lora_rank, H_idx*D_idx] col-sharded on out
        self._idx_wk = shard_in(w["indexer.wk"])  # [dim, D_idx] sharded on dim
        self._idx_wproj = shard_in(w["indexer.weights_proj"])  # [dim, H_idx] sharded on dim
        self._idx_knorm_w = repl(w["indexer.k_norm"])  # [D_idx]
        self._idx_knorm_b = repl(w["indexer.k_norm_bias"])  # [D_idx]

    def _device_rope_pe(self, x: ttnn.Tensor, glob: int, start_pos: int) -> ttnn.Tensor:
        """On-device non-interleaved RoPE on the rope half (first 64) of the last dim
        (backlog 19 / issue #4). x [1, n_heads, glob, D_idx]; cos/sin sliced to this
        chunk's global positions. Returns [1, n_heads, glob, D_idx]."""
        h = x.shape[1]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, glob, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, glob, self.index_args.index_head_dim])
        cos = ttnn.slice(self._idx_cos, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        sin = ttnn.slice(self._idx_sin, [0, 0, start_pos, 0], [1, 1, start_pos + glob, 64])
        pe = ttnn.experimental.rotary_embedding_hf(
            pe, cos, sin, is_decode_mode=False, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
        )
        return ttnn.concat([pe, nope], dim=-1)

    def _indexer_write_k(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int):
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
        self._index_kbuf = (
            k if start_pos == 0 or self._index_kbuf is None else ttnn.concat([self._index_kbuf, k], dim=2)
        )

    def _indexer_topk(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int = 0) -> ttnn.Tensor:
        """Top-k indices [1, 1, glob, k] over the device index-key cache. Fully on
        device (backlog 6 + 19): stems, RoPE, cache, logits, topk — no host."""
        a = self.index_args
        glob = seq_len * self.sp_factor  # global query/key count this chunk
        end_pos = start_pos + glob
        self._indexer_write_k(hidden_states, seq_len, start_pos)

        # Q stem: reuse v3 q_a latent (qr), then indexer wq_b — all on device.
        qr = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_a_proj", seq_len),
        )
        qr = self._tp_rs_ag(qr)
        qr = ttnn.rms_norm(
            qr,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        q = ttnn.linear(
            qr,
            self._idx_wq_b,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, S/sp, H_idx*D_idx]
        q = self._sp_all_gather(q, dim=2)  # [1, 1, glob, (H_idx/tp)*D_idx] full-seq, head-sharded on tp
        heads_local = a.index_n_heads // self.tp_factor  # this chip's indexer heads (col-parallel wq_b)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads_local, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, H_idx/tp, glob, D_idx]
        q_dev = self._device_rope_pe(q, glob, start_pos)

        # weights_proj: device stem -> reduce-scatter the H_idx heads across tp (each chip keeps the
        # reduced H_idx/tp slice matching its wq_b heads) -> SP gather -> scale -> [1, 1, glob, H_idx/tp].
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wts = self._tp_rs_ag(wts, rs_only=True)  # reduce-scatter on the head dim -> this chip's heads
        wts = self._sp_all_gather(wts, dim=2)
        wts = ttnn.multiply(wts, a.index_n_heads**-0.5 * self.scale)

        # Causality is fused inside indexer_score (future columns -> -inf from chunk_start_idx),
        # so no triu mask add here. Each chip scores only its H_idx/tp heads -> a PARTIAL logit
        # (the head-sum is separable; the -inf mask is head-independent so it survives the sum).
        # HB=0 keeps all H_idx/tp heads resident (fits L1 for tp>=2, i.e. <=32 heads/chip); tp=1 has
        # all 64 heads on one chip and must head-stream (HB=16). See INDEXER_OP.md "head residency".
        hb = 0 if self.tp_factor > 1 else 16
        cfg = ops.indexer_program_config(end_pos, head_group=hb)
        logits = ops.indexer_logits(q_dev, self._index_kbuf, wts, chunk_start_idx=start_pos, program_config=cfg)
        # All-reduce(SUM) the partial logits over tp -> full head-summed logit before top-k. The op emits
        # ROW_MAJOR; _tp_rs_ag (reduce_scatter+all_gather) runs in TILE, so round-trip the layout.
        if self.tp_factor > 1:
            # The op emits ROW_MAJOR; round-trip to TILE for the all-reduce. Passing RM straight to the
            # CCL is correct but ~10 ms slower — ttnn's RM reduce_scatter/all_gather tilize-with-padding
            # internally and add RM concats, costing more than this explicit ~6 ms tilize/untilize.
            logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT)
            logits = self._tp_rs_ag(logits)  # RS+AG over tp_axis == all-reduce SUM (reduce accumulates fp32)
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        return ops.topk_indices(logits, min(self.index_args.index_topk, end_pos))

    def forward(self, hidden_states, rope_tensors, kvpe_cache, **kwargs):
        seq_len_local = hidden_states.shape[2]
        kv_isl = kwargs.get("kv_actual_isl")
        end_pos = (kv_isl or 0) + seq_len_local * self.sp_factor
        if end_pos <= self.index_args.index_topk or not self._has_indexer:
            if self._has_indexer and kv_isl is not None:
                self._indexer_write_k(hidden_states, seq_len_local, kv_isl)
            return super().forward(hidden_states, rope_tensors, kvpe_cache, **kwargs)

        indices = self._indexer_topk(hidden_states, seq_len_local, start_pos=kv_isl or 0)
        return self._dsa_forward(hidden_states, rope_tensors, kvpe_cache, indices, **kwargs)

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

    def _dsa_forward(
        self,
        hidden_states,
        rope_tensors,
        kvpe_cache,
        indices,
        cache_layer_idx=0,
        kv_actual_isl=None,
        cache_user_id=0,
        **_,
    ):
        """v3 forward with ring SDPA replaced by sparse_mla (latent V); single-shot or chunked."""
        seq_len_local = hidden_states.shape[2]
        num_heads_local = self.num_heads // self.tp_factor
        is_chunked = kv_actual_isl is not None
        start_pos = kv_actual_isl or 0

        # Q stem (v3 forward lines 683..758)
        q = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_a_proj", seq_len_local),
        )
        q = self._tp_rs_ag(q)
        q = ttnn.rms_norm(
            q,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        q = ttnn.linear(
            q,
            self.q_b_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_b_proj", seq_len_local),
        )
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=num_heads_local, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, num_heads_local, seq_len_local, self.qk_nope_head_dim])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope_head_dim], [1, num_heads_local, seq_len_local, self.qk_head_dim])
        ttnn.deallocate(q)
        q_nope = ttnn.linear(
            q_nope,
            self.wkv_b1_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b1", seq_len_local),
        )
        q_rope = self._apply_rope(q_rope, rope_tensors, kv_actual_isl)
        q = ttnn.concat([q_nope, q_rope], dim=-1)  # absorbed [1, H, S, 576]
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope)

        # KV stem (v3 forward lines 761..829)
        kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local),
        )
        if self.tp_factor > 1:
            kv = self._tp_ag_reduce(kv)
        kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, seq_len_local, self.kv_lora_rank])
        kv_rope = ttnn.slice(
            kv, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim]
        )
        ttnn.deallocate(kv)
        kv_nope = ttnn.rms_norm(
            kv_nope,
            weight=self.kv_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        kv_rope = self._apply_rope(kv_rope, rope_tensors, kv_actual_isl)
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)
        ttnn.deallocate(kv_rope)
        kvpe_b8 = ttnn.typecast(kvpe, dtype=ttnn.bfloat8_b)
        end_pos = start_pos + seq_len_local * self.sp_factor

        # Build the full-T latent prefix the sparse op consumes — [1, 1, end_pos, 576] bf16
        # ROW_MAJOR, replicated across the mesh — entirely ON DEVICE (no host round-trip; backlog 9).
        if not is_chunked:
            # Single-shot: the live kvpe IS the whole sequence (natural order, contiguous SP shard).
            ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, kvpe_b8, cache_layer_idx)
            gathered = self._sp_all_gather(kvpe, dim=2)  # [1,1,T,576] bf16 TILE, replicated, natural
            kvpe_dev = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
            if self.sp_factor > 1:
                ttnn.deallocate(gathered)
        else:
            # Chunked: the prefix lives in the BLOCK-CYCLIC cache; gather + un-rotate on device
            # (device equivalent of the former ConcatMesh2dToTensor + blockcyclic_positions readback).
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache,
                kvpe_b8,
                slot_idx=cache_user_id,
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                kv_actual_global=kv_actual_isl,
                cluster_axis=self.sp_axis,
            )
            kvpe_dev = self._gather_kvpe_prefix(kvpe_cache, seq_len_local, end_pos, cache_user_id, cache_layer_idx)
        ttnn.deallocate(kvpe_b8)
        ttnn.deallocate(kvpe)

        # Sparse attention over selected latents (replaces ring SDPA); latent out [1,H,S,512].
        # q SP×TP-sharded, kvpe full-T replicated on device, indices full (replicated).
        attn_out = ops.sparse_mla(
            q, kvpe_dev, indices, self.scale, start_pos=start_pos, sp_axis=self.sp_axis, tp_axis=self.tp_axis
        )
        ttnn.deallocate(kvpe_dev)
        ttnn.deallocate(q)

        # wkv_b2 AFTER attention (chunked-path style), then v3 epilogue
        attn_out = ttnn.linear(
            attn_out,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )
        out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(
            out,
            self.o_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("o_proj", seq_len_local),
        )
        if self.tp_factor > 1:
            out = self._tp_rs_ag(out, rs_only=True)  # v3 epilogue: RS only, output stays TP-sharded
        return out

    def _gather_kvpe_prefix(self, kvpe_cache, seq_len_local, end_pos, cache_user_id, cache_layer_idx):
        """On-device read-back of the chunked KVPE prefix for sparse attention. The cache is
        bf8 / TILE / ND-sharded / block-cyclic across SP; the sparse op wants the full prefix
        bf16 / ROW_MAJOR / replicated / natural order. Pipeline (all on device — replaces the
        former host ConcatMesh2dToTensor + blockcyclic_positions read): ND→interleaved, SP
        all-gather to full-T (no-op at sp==1), select this user/layer slot, bf8→bf16, TILE→RM,
        un-rotate block-cyclic→natural, trim to end_pos. Returns [1, 1, end_pos, 576] bf16 RM."""
        cache_i = ttnn.to_memory_config(kvpe_cache, ttnn.DRAM_MEMORY_CONFIG)  # ND_SHARDED → INTERLEAVED
        full = self._sp_all_gather(cache_i, dim=2)  # → [B, 1, seq_len_cache, 576] replicated, block-cyclic
        if self.sp_factor > 1:
            ttnn.deallocate(cache_i)
        if full.shape[0] > 1:  # user-major slot select (no-op for the single-slot cache)
            slot = cache_user_id * self.layer_num + cache_layer_idx
            sel = ttnn.slice(full, [slot, 0, 0, 0], [slot + 1, 1, full.shape[2], full.shape[3]])
            ttnn.deallocate(full)
            full = sel
        full16 = ttnn.typecast(full, ttnn.bfloat16)
        ttnn.deallocate(full)
        full_rm = ttnn.to_layout(full16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(full16)
        out = self._unrotate_blockcyclic(full_rm, seq_len_local, end_pos)
        ttnn.deallocate(full_rm)
        return out

    def _unrotate_blockcyclic(self, t, seq_len_local, end_pos):
        """Reorder a gathered full-cache tensor from the cache's block-cyclic SP layout into
        natural token order, on device — the inverse of deepseek_v3_d_p ...mla.utils
        .blockcyclic_positions. t is [1, 1, seq_len_cache, 576] ROW_MAJOR (block-cyclic);
        returns [1, 1, end_pos, 576] natural order. Identity order at sp==1."""
        sp = self.sp_factor
        seq_len_cache, d = t.shape[2], t.shape[3]
        chunk_local = seq_len_local  # = chunk_size_global / sp
        slabs = seq_len_cache // (sp * chunk_local)  # chunks written into the cache
        if sp == 1:
            return ttnn.slice(t, (0, 0, 0, 0), (1, 1, end_pos, d))  # identity order
        # Reorder block-cyclic [chip, slab, off] → natural [slab, chip, off]. A reshape+permute leaves
        # the result physically NON-contiguous (a strided view): to_torch honours the strides, but the
        # sparse_sdpa reader consumes raw physical order and silently reads garbage. Concatenating
        # contiguous per-(slab,chip) slabs materialises a fresh contiguous tensor instead.
        seq_local_cache = slabs * chunk_local  # rows held by one chip
        blocks = [
            ttnn.slice(
                t,
                (0, 0, c * seq_local_cache + s * chunk_local, 0),
                (1, 1, c * seq_local_cache + (s + 1) * chunk_local, d),
            )
            for s in range(slabs)
            for c in range(sp)
        ]
        flat = ttnn.concat(blocks, dim=2)
        for b in blocks:
            ttnn.deallocate(b)
        out = flat if end_pos == seq_len_cache else ttnn.slice(flat, (0, 0, 0, 0), (1, 1, end_pos, d))
        return out

    def _tp_rs_ag(self, t, rs_only=False):
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

    def _tp_ag_reduce(self, t):
        t = ttnn.experimental.all_gather_async(
            t,
            dim=1,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )
        return ttnn.experimental.fast_reduce_nc(
            t, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
        )
