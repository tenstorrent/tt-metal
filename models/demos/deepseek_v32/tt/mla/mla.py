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
from models.demos.deepseek_v32.reference_cpu.utils import apply_rotary_emb, precompute_freqs_cis
from models.demos.deepseek_v32.tt import ops

INDEXER_WEIGHT_NAMES = ("indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.k_norm_bias", "indexer.weights_proj")


class ttMLA(_ttMLAv3):
    """V3.2 MLA with DSA. Dense passthrough of v3 at seq <= index_topk."""

    def __init__(self, config, weights, mesh_device, **kwargs):
        # Indexer weights are v32-only — pop before v3 sees them, upload to device
        # after super().__init__ (which sets mesh_device / tp_axis / tp_factor).
        idx_host = {n: weights.pop(f"{n}.weight") for n in INDEXER_WEIGHT_NAMES if f"{n}.weight" in weights}
        self.index_args = ModelArgs(max_batch_size=1)
        # Host indexer K-cache (chunked prefill): chunks only carry their own
        # hidden, but logits need every prior key. [max_seq, D_idx] per layer.
        self._index_k_cache = torch.zeros(
            self.index_args.max_seq_len, self.index_args.index_head_dim, dtype=torch.bfloat16
        )
        # Host bf16 KVPE mirror (chunked sparse fallback): avoids double bf8
        # quantization in the host workaround; the fused op will read the bf8
        # cache like ring_mla (agreement: v3 cache format).
        self._kvpe_mirror = torch.zeros(
            self.index_args.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim, dtype=torch.bfloat16
        )
        super().__init__(config, weights, mesh_device, **kwargs)
        self._has_indexer = bool(idx_host)
        if self._has_indexer:
            self._upload_indexer_weights(idx_host)

    def _upload_indexer_weights(self, w):
        """Indexer weights → device (backlog 6: stems on device, replicated across TP).

        wk / weights_proj contract over `dim` (hidden is TP-sharded on dim), so they
        are uploaded transposed and sharded on that contraction axis → matmul yields
        per-chip partials reduced by _tp_rs_ag. wq_b consumes the full q-latent and
        k_norm runs on the reduced 128-wide key, so both are replicated.
        """

        def repl(t):
            return ttnn.from_torch(
                t.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        def shard_in(t):  # t [out, in] -> device [in, out] sharded on `in` across tp_axis
            dims = [None, None]
            dims[self.tp_axis] = 0
            return ttnn.from_torch(
                t.T.contiguous().to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
                ),
            )

        self._idx_wq_b = repl(w["indexer.wq_b"].T)  # [q_lora_rank, H_idx*D_idx]
        self._idx_wk = shard_in(w["indexer.wk"])  # [dim, D_idx] sharded on dim
        self._idx_wproj = shard_in(w["indexer.weights_proj"])  # [dim, H_idx] sharded on dim
        self._idx_knorm_w = repl(w["indexer.k_norm"])  # [D_idx]
        self._idx_knorm_b = repl(w["indexer.k_norm_bias"])  # [D_idx]

    def _host_rope_pe(self, t_host, seq_len, start_pos):
        """Non-interleaved RoPE on the rope half of the last dim (F1, host fallback).
        t_host [..., S, D_idx] in reference layout ([S,H,D] or [S,D]); returns same shape."""
        freqs = precompute_freqs_cis(self.index_args)[start_pos : start_pos + seq_len]
        pe, nope = t_host[..., :64], t_host[..., 64:]
        if pe.dim() == 3:  # q: [S, H, 64] -> reference wants [B, S, H, rope]
            pe = apply_rotary_emb(pe.unsqueeze(0), freqs, interleaved=False).squeeze(0)
        else:  # k: [S, 64] -> [B, S, 1, rope]
            pe = apply_rotary_emb(pe.reshape(1, seq_len, 1, 64), freqs, interleaved=False).reshape(seq_len, 64)
        return torch.cat([pe, nope], dim=-1)

    def _indexer_write_k(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int):
        """Device K stem (wk + TP all-reduce + k_norm), host rope, write host K-cache.
        Runs on EVERY chunk — dense-passthrough included — else later DSA chunks score
        against zero keys for the early prefix."""
        k = ttnn.linear(
            hidden_states,
            self._idx_wk,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # per-chip partial [1, 1, S, D_idx]
        k = self._tp_rs_ag(k)  # all-reduce over TP
        k = ttnn.layer_norm(
            k,
            weight=self._idx_knorm_w,
            bias=self._idx_knorm_b,
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        kh = ttnn.to_torch(ttnn.get_device_tensors(k)[0])[0, 0]  # replicated -> [S, D_idx]
        self._index_k_cache[start_pos : start_pos + seq_len] = self._host_rope_pe(kh, seq_len, start_pos).to(
            torch.bfloat16
        )

    def _indexer_topk(self, hidden_states: ttnn.Tensor, seq_len: int, start_pos: int = 0) -> ttnn.Tensor:
        """Top-k indices [1, 1, S, k] over cached prefix + this chunk. Stems on device
        (backlog 6); only the non-interleaved RoPE stays on host (F1)."""
        a = self.index_args
        end_pos = start_pos + seq_len
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
        )  # [1, 1, S, H_idx*D_idx]
        qh = ttnn.to_torch(ttnn.get_device_tensors(q)[0]).reshape(seq_len, a.index_n_heads, a.index_head_dim)
        qh = self._host_rope_pe(qh, seq_len, start_pos)  # [S, H, D]

        # weights_proj: device stem + all-reduce + scale -> [1, 1, S, H_idx]
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wts = self._tp_rs_ag(wts)
        wts = ttnn.multiply(wts, a.index_n_heads**-0.5 * self.scale)

        dev = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        q_dev = dev(qh.permute(1, 0, 2).reshape(1, a.index_n_heads, seq_len, a.index_head_dim))
        k_dev = dev(self._index_k_cache[:end_pos].reshape(1, 1, end_pos, a.index_head_dim))
        logits = ops.indexer_logits(q_dev, k_dev, wts)
        mask = dev(torch.full((1, 1, seq_len, end_pos), float("-inf")).triu_(start_pos + 1))
        return ops.topk_indices(ttnn.add(logits, mask), min(self.index_args.index_topk, end_pos))

    def forward(self, hidden_states, rope_tensors, kvpe_cache, **kwargs):
        seq_len_local = hidden_states.shape[2]
        kv_isl = kwargs.get("kv_actual_isl")
        end_pos = (kv_isl or 0) + seq_len_local * self.sp_factor
        if end_pos <= self.index_args.index_topk or not self._has_indexer:
            if self._has_indexer and kv_isl is not None:
                self._indexer_write_k(hidden_states, seq_len_local, kv_isl)
            return super().forward(hidden_states, rope_tensors, kvpe_cache, **kwargs)
        if self.sp_factor != 1:
            raise NotImplementedError("DSA bring-up: sp=1 only (status.md step 4)")

        indices = self._indexer_topk(hidden_states, seq_len_local, start_pos=kv_isl or 0)
        return self._dsa_forward(hidden_states, rope_tensors, kvpe_cache, indices, **kwargs)

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
        if not is_chunked:
            ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, kvpe_b8, cache_layer_idx)
            kvpe_prefix = kvpe
        else:
            # Same cache write as v3 _chunked_attn; attention reads the populated prefix.
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache,
                kvpe_b8,
                slot_idx=cache_user_id,
                layer_idx=cache_layer_idx,
                num_layers=self.layer_num,
                kv_actual_global=kv_actual_isl,
                cluster_axis=self.sp_axis,
            )
            slot = cache_user_id * self.layer_num + cache_layer_idx
            prefix = ttnn.to_torch(ttnn.get_device_tensors(kvpe_cache)[0])[
                slot : slot + 1, :, : start_pos + seq_len_local
            ]
            kvpe_prefix = ttnn.from_torch(
                prefix.to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        # Sparse attention over selected latents (replaces ring SDPA); latent out [1,H,S,512]
        attn_out = ops.sparse_mla(q, kvpe_prefix, indices, self.scale, start_pos=start_pos)

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
