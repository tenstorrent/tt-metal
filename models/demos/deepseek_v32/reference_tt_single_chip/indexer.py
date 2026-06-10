# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn (single-device) port of the DeepSeek-V3.2 Indexer (the "lightning indexer"
of DeepSeek Sparse Attention).  Mirrors ``reference_cpu.model.IndexerCPU`` on its
*functional* path (``use_fp8_path=False``): the orthogonal Hadamard transform and
fp8 quantization are dropped (precision-only; MLA_LAYER.md §A.6), so the score is
computed directly as ``sum_h relu(q_h · k) * weights_h``.

CPU fallbacks (no clean ttnn equivalent, spec §6/§10): the non-interleaved RoPE
and the final ``topk`` selection.  The continuous ``index_score`` — the part that
matters for PCC — is computed entirely on device.
"""

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.reference_tt_single_chip.utils import (
    apply_noninterleaved_rope_host,
    convert_indexer_weights,
    default_compute_kernel_config,
)


class ttIndexer:
    def __init__(self, args, state_dict: dict, mesh_device: ttnn.MeshDevice, layer_idx: int = 0, cache_path=None):
        self.args = args
        self.mesh_device = mesh_device
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.ln_eps = 1e-6
        self.freqs_cis = precompute_freqs_cis(args)

        cache_path = Path(cache_path) if cache_path is not None else None
        self.w = convert_indexer_weights(state_dict, mesh_device, layer_idx, cache_path)

        self.compute_kernel_config = default_compute_kernel_config(mesh_device)

    def _linear(self, x, weight):
        return ttnn.linear(x, weight, compute_kernel_config=self.compute_kernel_config)

    def forward(self, x: ttnn.Tensor, qr: ttnn.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x:  ``[B, 1, S, dim]`` input features.
            qr: ``[B, 1, S, q_lora]`` MLA query latent (post q_norm).
            start_pos: absolute position base (rope) / cache write offset.
            mask: optional additive ``[S, end_pos]`` causal mask (host tensor).
        Returns:
            (topk_indices ``[B, S, min(topk, end_pos)]``, index_score ``[B, S, end_pos]``)
            as host tensors, matching ``IndexerCPU``.
        """
        b, _, s, _ = x.shape
        Hi, D, r = self.n_heads, self.head_dim, self.rope_head_dim
        end = start_pos + s

        # ----- query: rope FIRST then nope (non-interleaved rope) -----
        q = self._linear(qr, self.w["wq_b"])  # [B,1,S,Hi*D]
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=Hi, num_kv_heads=0, transpose_k_heads=False)
        q_pe = ttnn.slice(q, [0, 0, 0, 0], [b, Hi, s, r])
        q_nope = ttnn.slice(q, [0, 0, 0, r], [b, Hi, s, D])
        ttnn.deallocate(q)
        q_pe = apply_noninterleaved_rope_host(q_pe, self.freqs_cis, start_pos, self.mesh_device)
        q = ttnn.concat([q_pe, q_nope], dim=-1)  # [B,Hi,S,D]
        ttnn.deallocate(q_pe)
        ttnn.deallocate(q_nope)

        # ----- key: single head, LayerNorm, rope first -----
        k = self._linear(x, self.w["wk"])  # [B,1,S,D]
        k = ttnn.layer_norm(
            k,
            weight=self.w["k_norm_weight"],
            bias=self.w["k_norm_bias"],
            epsilon=self.ln_eps,
            compute_kernel_config=self.compute_kernel_config,
        )
        k_pe = ttnn.slice(k, [0, 0, 0, 0], [b, 1, s, r])
        k_nope = ttnn.slice(k, [0, 0, 0, r], [b, 1, s, D])
        ttnn.deallocate(k)
        k_pe = apply_noninterleaved_rope_host(k_pe, self.freqs_cis, start_pos, self.mesh_device)
        k = ttnn.concat([k_pe, k_nope], dim=-1)  # [B,1,S,D]
        ttnn.deallocate(k_pe)
        ttnn.deallocate(k_nope)

        # ----- per-head weights: weights_proj(x) * Hi**-0.5 * softmax_scale -----
        w = self._linear(x, self.w["weights_proj"])  # [B,1,S,Hi]
        w = ttnn.multiply(w, self.n_heads**-0.5 * self.softmax_scale)
        w = ttnn.permute(w, (0, 3, 2, 1))  # [B,Hi,S,1]

        # ----- index_score[b,s,t] = sum_h relu(q_h . k) * w_h  (k single-head) -----
        k_rep = ttnn.repeat(k, ttnn.Shape([1, Hi, 1, 1]))  # [B,Hi,S,D] (head broadcast)
        ttnn.deallocate(k)
        logits = ttnn.matmul(q, ttnn.transpose(k_rep, -2, -1), compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q)
        ttnn.deallocate(k_rep)
        logits = ttnn.relu(logits)
        logits = ttnn.multiply(logits, w)  # broadcast over T -> [B,Hi,S,T]
        index_score = ttnn.sum(logits, dim=1)  # sum over heads -> [B,S,T]
        ttnn.deallocate(logits)

        index_score = (
            ttnn.to_torch(index_score, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[:b]
            .reshape(b, s, end)
            .float()
        )

        # ----- mask + topk (CPU FALLBACK, spec §6/§10) -----
        if mask is not None:
            index_score = index_score + mask
        topk_indices = index_score.topk(min(self.index_topk, end), dim=-1)[1]
        return topk_indices, index_score
