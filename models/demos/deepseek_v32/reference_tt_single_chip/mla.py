# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn (single-device) port of the DeepSeek-V3.2 MLA layer.

Mirrors ``reference_cpu.model.MLACPU`` op-for-op with ttnn ops (spec §4).
Single device only (``world_size == 1``): plain ``ttnn.linear`` for the
column/row-parallel projections, replicated weights, no CCL.  Attention is
computed manually (matmul → scale+mask → softmax → matmul) rather than via a
fused SDPA op, so the DSA additive index mask is added to the scores before
softmax and the path maps directly onto the reference einsums (perf is out of
scope, §9).  (The qk=192/v=128 dims are *not* the reason — SDPA supports a
separate ``head_dim_v``.)

The latent KV-cache fp8 precision simulation is dropped (the reference is run
with ``simulate_fp8=False`` for parity); the indexer's additive mask is a no-op
at the tested sequence lengths (S <= index_topk) and is wired in separately
(spec §8 step 4).
"""

import math
from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v32.reference_tt_single_chip.indexer import ttIndexer
from models.demos.deepseek_v32.reference_tt_single_chip.utils import (
    RopeTables,
    apply_interleaved_rope,
    convert_mla_weights,
    default_compute_kernel_config,
    replicate_to_device,
)


class ttMLA:
    def __init__(
        self,
        args,
        state_dict: dict,
        mesh_device: ttnn.MeshDevice,
        layer_idx: int = 0,
        cache_path=None,
        with_indexer: bool = False,
    ):
        self.args = args
        self.mesh_device = mesh_device
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.rms_eps = 1e-6

        # softmax scale with YaRN mscale correction (matches MLACPU).
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        cache_path = Path(cache_path) if cache_path is not None else None
        self.w = convert_mla_weights(state_dict, args, mesh_device, layer_idx, cache_path)

        # Nested sparse-attention selector.  Off by default: at the tested
        # sequence lengths (S <= index_topk) its additive mask is all zero, so it
        # is a no-op on the MLA output and only adds compute.  Enabled (with a
        # small index_topk) to exercise the sparse-mask wiring (spec §8 step 4).
        self.index_topk = args.index_topk
        self.indexer = ttIndexer(args, state_dict, mesh_device, layer_idx, cache_path) if with_indexer else None

        # Latent KV cache.  The latent (post-kv_norm kv, and rope key k_pe) is
        # computed on device; the cache *storage/indexing* is kept on the host
        # (CPU FALLBACK): the ttnn cache ops require tile-aligned (mult-of-32)
        # write offsets, which the prefill(P-1)+decode(1) equivalence test
        # (start_pos=7) violates.  A bf16 host round-trip is lossless, so this
        # is numerically exact; perf is out of scope (spec §9).
        self.kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, dtype=torch.bfloat16)
        self.pe_cache = torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16)

        self.compute_kernel_config = default_compute_kernel_config(mesh_device)

    def _rms(self, x, weight):
        return ttnn.rms_norm(x, weight=weight, epsilon=self.rms_eps, compute_kernel_config=self.compute_kernel_config)

    def _linear(self, x, weight):
        return ttnn.linear(x, weight, compute_kernel_config=self.compute_kernel_config)

    def _matmul(self, a, b):
        return ttnn.matmul(a, b, compute_kernel_config=self.compute_kernel_config)

    def _to_torch(self, t: ttnn.Tensor) -> torch.Tensor:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))

    def _to_device(self, t: torch.Tensor) -> ttnn.Tensor:
        return replicate_to_device(t, self.mesh_device)

    def _write_cache(self, kv_nope: ttnn.Tensor, k_pe: ttnn.Tensor, start_pos: int, b: int, s: int):
        """Store the (device-computed) latent into the host cache slot [start:end]."""
        end = start_pos + s
        kv_host = self._to_torch(kv_nope)[:b].reshape(b, s, self.kv_lora_rank)
        pe_host = self._to_torch(k_pe)[:b].reshape(b, s, self.qk_rope_head_dim)
        self.kv_cache[:b, start_pos:end] = kv_host
        self.pe_cache[:b, start_pos:end] = pe_host

    def _read_cache(self, b: int, end: int):
        """Upload the valid cache range [0:end] back to device as [B,1,end,*]."""
        kv = self._to_device(self.kv_cache[:b, :end].reshape(b, 1, end, self.kv_lora_rank))
        pe = self._to_device(self.pe_cache[:b, :end].reshape(b, 1, end, self.qk_rope_head_dim))
        return kv, pe

    def _index_mask(self, x, qr, start_pos, s, b, is_prefill):
        """
        Run the indexer and turn its top-k selection into an additive
        ``{0, -inf}`` mask ``[B, 1, S, end]`` (broadcast over heads), matching
        MLACPU.  Returns None when no indexer is attached.  At S <= index_topk
        the selection is every position, so the mask is all zero.

        CPU FALLBACK (spec §6/§10): the scatter of top-k indices into the
        ``{0, -inf}`` mask is built on the host (no clean ttnn topk/scatter over
        the key axis) and uploaded; ``ttIndexer`` likewise returns host
        top-k indices.  The continuous index_score is computed on device.
        """
        if self.indexer is None:
            return None
        end = start_pos + s
        causal = None
        if is_prefill:  # indexer scores only past keys, like the reference prefill mask
            qpos = torch.arange(start_pos, end).unsqueeze(1)
            kpos = torch.arange(end).unsqueeze(0)
            causal = torch.where(kpos <= qpos, 0.0, float("-inf"))  # [S, end]
        topk_indices, _ = self.indexer.forward(x, qr, start_pos, mask=causal)  # [B,S,topk]
        idx_mask = torch.full((b, s, end), float("-inf")).scatter_(-1, topk_indices, 0.0)
        return self._to_device(idx_mask.reshape(b, 1, s, end))  # additive over heads

    def forward(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        rope: RopeTables,
        causal_mask: Optional[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """
        Args:
            x: ``[B, 1, S, dim]`` (tiled, bf16).
            start_pos: absolute cache write offset (rope position base).
            rope: ``RopeTables`` providing the per-call cos/sin/trans tensors.
            causal_mask: additive ``[1, 1, S, S]`` mask for prefill, else None.
        Returns:
            ``[B, 1, S, dim]``.
        """
        b, _, s, _ = x.shape
        H = self.n_heads
        rope_t = rope.rope_tensors(start_pos, s)

        # ----- shared front-end: query latent path -----
        qr = self._rms(self._linear(x, self.w["wq_a"]), self.w["q_norm"])  # [B,1,S,q_lora]
        q = self._linear(qr, self.w["wq_b"])  # [B,1,S,H*qk_head_dim]
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=H, num_kv_heads=0, transpose_k_heads=False
        )  # [B,H,S,qk_head_dim]
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [b, H, s, self.qk_nope_head_dim])  # [B,H,S,nope]
        q_pe = ttnn.slice(q, [0, 0, 0, self.qk_nope_head_dim], [b, H, s, self.qk_head_dim])  # [B,H,S,rope]
        ttnn.deallocate(q)
        q_pe = apply_interleaved_rope(q_pe, rope_t)

        # ----- shared front-end: kv latent path -----
        kv = self._linear(x, self.w["wkv_a"])  # [B,1,S,kv_lora+rope]
        kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [b, 1, s, self.kv_lora_rank])
        k_pe = ttnn.slice(kv, [0, 0, 0, self.kv_lora_rank], [b, 1, s, self.kv_lora_rank + self.qk_rope_head_dim])
        ttnn.deallocate(kv)
        kv_nope = self._rms(kv_nope, self.w["kv_norm"])  # [B,1,S,kv_lora]
        k_pe = apply_interleaved_rope(k_pe, rope_t)  # [B,1,S,rope]

        # Write the latent into the (host) cache on both paths, so decode reads
        # exactly what prefill stored (cache-correctness invariant, spec §4.4).
        self._write_cache(kv_nope, k_pe, start_pos, b, s)

        # Sparse-attention additive mask from the indexer (None if not attached
        # or all-zero when S <= index_topk).
        index_mask = self._index_mask(x, qr, start_pos, s, b, causal_mask is not None)

        if causal_mask is not None:
            x_attn = self._prefill_attn(q_nope, q_pe, kv_nope, k_pe, causal_mask, index_mask, b, s, H)
        else:
            x_attn = self._decode_attn(q_nope, q_pe, index_mask, start_pos + s, b, H)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_pe)

        # Merge heads -> [B, 1, S, H*v].  (nlp_concat_heads overflows L1 for 128
        # heads; a transpose+reshape is equivalent and fine — perf is out of scope.)
        x_attn = ttnn.transpose(x_attn, 1, 2)  # [B,S,H,v]
        x_attn = ttnn.reshape(x_attn, (b, 1, s, H * self.v_head_dim))  # [B,1,S,H*v]
        out = self._linear(x_attn, self.w["wo"])  # [B,1,S,dim]
        return out

    def _prefill_attn(self, q_nope, q_pe, kv_nope, k_pe, causal_mask, index_mask, b, s, H):
        """MHA: materialize per-head K and V via wkv_b, dense scaled-dot attention."""
        q = ttnn.concat([q_nope, q_pe], dim=-1)  # [B,H,S,qk_head_dim]
        kv_b = self._linear(kv_nope, self.w["wkv_b"])  # [B,1,S,H*(nope+v)]
        kv_b, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            kv_b, num_heads=H, num_kv_heads=0, transpose_k_heads=False
        )  # [B,H,S,nope+v]
        k_nope = ttnn.slice(kv_b, [0, 0, 0, 0], [b, H, s, self.qk_nope_head_dim])
        v = ttnn.slice(kv_b, [0, 0, 0, self.qk_nope_head_dim], [b, H, s, self.qk_nope_head_dim + self.v_head_dim])
        ttnn.deallocate(kv_b)
        k_pe_h = ttnn.repeat(k_pe, ttnn.Shape([1, H, 1, 1]))  # broadcast shared rope key over heads
        k = ttnn.concat([k_nope, k_pe_h], dim=-1)  # [B,H,S,qk_head_dim]
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_pe_h)

        scores = self._matmul(q, ttnn.transpose(k, -2, -1))  # [B,H,S,T]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        scores = ttnn.multiply(scores, self.softmax_scale)
        scores = ttnn.add(scores, causal_mask)  # broadcast [1,1,S,T] over B,H
        if index_mask is not None:
            scores = ttnn.add(scores, index_mask)  # sparse {0,-inf} selection
        scores = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
        x_attn = self._matmul(scores, v)  # [B,H,S,v]
        ttnn.deallocate(scores)
        return x_attn

    def _decode_attn(self, q_nope, q_pe, index_mask, end, b, H):
        """MQA with wkv_b absorption: attend directly against the latent cache."""
        kv_cache, pe_cache = self._read_cache(b, end)  # [B,1,T,c], [B,1,T,rope]
        # ttnn.matmul broadcasts neither batch (dim0) nor heads (dim1) here, so
        # materialize both axes: repeat the absorption weights to batch B and the
        # (head-shared) latent cache to all H heads.  Cheap at these T; perf §9.
        wkv_b1 = ttnn.repeat(self.w["wkv_b1"], ttnn.Shape([b, 1, 1, 1]))  # [B,H,nope,c]
        wkv_b2 = ttnn.repeat(self.w["wkv_b2"], ttnn.Shape([b, 1, 1, 1]))  # [B,H,c,v]
        kv_cache = ttnn.repeat(kv_cache, ttnn.Shape([1, H, 1, 1]))  # [B,H,T,c]
        pe_cache = ttnn.repeat(pe_cache, ttnn.Shape([1, H, 1, 1]))  # [B,H,T,rope]

        # Absorb the nope half of wkv_b into the query -> width c.
        q_nope_abs = self._matmul(q_nope, wkv_b1)  # [B,H,1,c]
        scores = self._matmul(q_nope_abs, ttnn.transpose(kv_cache, -2, -1))  # [B,H,1,T]
        scores = ttnn.add(scores, self._matmul(q_pe, ttnn.transpose(pe_cache, -2, -1)))  # + q_pe . pe_cache
        ttnn.deallocate(q_nope_abs)
        scores = ttnn.multiply(scores, self.softmax_scale)
        if index_mask is not None:
            scores = ttnn.add(scores, index_mask)  # sparse {0,-inf} selection
        scores = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
        x_attn = self._matmul(scores, kv_cache)  # [B,H,1,c]
        ttnn.deallocate(scores)
        x_attn = self._matmul(x_attn, wkv_b2)  # absorb value half -> [B,H,1,v]
        return x_attn
