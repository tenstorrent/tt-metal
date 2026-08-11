# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-compatible reference for the DeepSeek V3.2 attention stack.

Mirrors the GPU reference (``DeepSeek-V3.2-Exp/inference/model.py``) but replaces
the tilelang CUDA kernels and the tensor-parallel ``Linear`` variants with plain
PyTorch so the layers run and can be inspected on CPU.

Contains:
  * ``ModelArgs``  — unified config (671B v3.2) for both layers.
  * Building blocks — ``Linear``, ``LayerNorm``, ``RMSNorm``.
  * ``IndexerCPU``  — the "lightning indexer" sparse-attention selector; its only
    output is ``topk_indices``.
  * ``MLACPU``      — Multi-Head Latent Attention; consumes the indexer's
    ``topk_indices`` and runs both the prefill (MHA) and decode (MQA-absorbed)
    paths.

This module is architecture only. CPU kernel equivalents and RoPE helpers live
in ``utils.py``; weight init + pretrained HF loading live in ``weights.py``; the
test harness in ``test_model.py``.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.utils import (
    act_quant_cpu,
    apply_rotary_emb,
    fp8_index_cpu,
    rotate_activation_cpu,
)

# block size for the fp8 quantization simulation (matches reference kernel.py)
BLOCK_SIZE = 128


@dataclass
class ModelArgs:
    """
    Unified model arguments for the Indexer + MLA, from config_671B_v3.2.json.

    Carries every field both layers read, so a single instance can be shared
    between ``MLACPU`` and its nested ``IndexerCPU``.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4  # 16384
    dim: int = 7168
    # MLA
    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # YaRN / RoPE
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # Quantization scale format ("ue8m0" -> power-of-two scales).
    scale_fmt: Optional[str] = "ue8m0"
    # Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    # Indexer RoPE convention: DeepSeek-V3.2 = non-interleaved (rotate_half);
    # GLM-5.1 = interleaved. (MLA RoPE is always interleaved.)
    index_rope_interleave: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# ===== Building blocks =====


class Linear(nn.Module):
    """
    Simple linear layer matching reference implementation.

    Note: Weights are allocated as empty tensors and must be initialized
    via initialize_weights() before use. The reference constructs all of its
    projections bias-free (bias=False), so bias defaults to off here.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Allocate empty tensors (matches reference implementation)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LayerNorm(nn.Module):
    """
    LayerNorm matching reference implementation (model.py:LayerNorm).

    Note: γ (weight) is initialized to 1 and β (bias) to 0. Computation is done
    in float32 and cast back to the input dtype, with eps=1e-6 to match the
    reference. Used by the Indexer's ``k_norm``.
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        # Initialize γ to 1, β to 0 (standard LayerNorm initialization), float32
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.weight.shape[0],), self.weight, self.bias, self.eps).type_as(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm matching the reference (model.py:RMSNorm).

    Computation is in float32 with eps=1e-6; ``weight`` (γ) initialized to ones.
    Unlike the Indexer's ``k_norm`` (a full LayerNorm with bias), MLA uses
    RMSNorm for ``q_norm`` and ``kv_norm``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


# ===== Indexer =====


class IndexerCPU(nn.Module):
    """
    CPU-compatible version of the Indexer layer (the "lightning indexer").

    Modified from the reference to use CPU-compatible operations. Its only output
    is ``topk_indices`` — the set of past positions each query is allowed to
    attend to.
    """

    def __init__(self, args: ModelArgs, use_fp8_path: bool = True):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.rope_interleave = args.index_rope_interleave  # GLM: True; DS: False
        self.q_lora_rank = args.q_lora_rank
        self.scale_fmt = args.scale_fmt
        # When False, skip the orthogonal Hadamard transform and the fp8
        # quant/dequant and score directly in bf16/fp32:
        # both are precision-only (Hadamard is orthogonal, so it does not change
        # q·k), so this is the functional path the ttnn port matches.
        self.use_fp8_path = use_fp8_path

        # Projections (bias-free, matching the reference indexer)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim**-0.5

        # Buffers (using bfloat16 instead of FP8 for CPU compatibility)
        self.register_buffer(
            "k_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // 128, dtype=torch.float32),
            persistent=False,
        )

        logger.info(f"Initialized IndexerCPU with {self.n_heads} heads, head_dim={self.head_dim}")

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of Indexer layer.

        Args:
            x: Input features [B, L, dim]
            qr: Query representation [B, L, q_lora_rank]
            start_pos: Starting position in cache
            freqs_cis: Rotary embedding frequencies
            mask: Optional attention mask

        Returns:
            (topk_indices, index_score)
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        logger.info(f"Indexer forward: bsz={bsz}, seqlen={seqlen}, start_pos={start_pos}")

        # Query projection
        q = self.wq_b(qr)  # [B, L, n_heads * head_dim]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # [B, L, H, D]

        # Split RoPE and non-RoPE parts
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # Apply RoPE (DS: non-interleaved / GLM: interleaved — ModelArgs.index_rope_interleave)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=self.rope_interleave)

        # Concatenate back
        q = torch.cat([q_pe, q_nope], dim=-1)  # [B, L, H, D]

        # Key projection
        k = self.wk(x)  # [B, L, D]
        k = self.k_norm(k)  # [B, L, D]

        # Split and apply RoPE to key
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=self.rope_interleave).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)  # [B, L, D]

        if self.use_fp8_path:
            # Hadamard transform
            q = rotate_activation_cpu(q)
            k = rotate_activation_cpu(k)

            # FP8 quantization (CPU version using bfloat16)
            q_fp8, q_scale = act_quant_cpu(q, block_size=BLOCK_SIZE, scale_fmt=self.scale_fmt)
            k_fp8, k_scale = act_quant_cpu(k, block_size=BLOCK_SIZE, scale_fmt=self.scale_fmt)

            # Update cache
            self.k_cache[:bsz, start_pos:end_pos] = k_fp8
            self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

            # Compute weights
            weights = self.weights_proj(x.float())  # [B, L, H]
            weights = weights * (self.n_heads**-0.5)  # Scale by 1/sqrt(n_heads)
            weights = weights.unsqueeze(-1)  # [B, L, H, 1]
            weights = weights * q_scale * self.softmax_scale  # [B, L, H, 1]

            # Compute index scores using FP8 index operation
            index_score = fp8_index_cpu(
                q_fp8,  # [B, L, H, D]
                weights.squeeze(-1),  # [B, L, H]
                self.k_cache[:bsz, :end_pos],  # [B, C, D]
                self.k_scale_cache[:bsz, :end_pos].squeeze(-1),  # [B, C]
            )
        else:
            # Clean functional path (no Hadamard, no fp8): the dequant scales are
            # identity, so fold only weights_proj * Hi**-0.5 * softmax_scale into
            # the per-head weights and pass unit key scales.
            self.k_cache[:bsz, start_pos:end_pos] = k
            weights = self.weights_proj(x.float()) * (self.n_heads**-0.5) * self.softmax_scale  # [B, L, H]
            index_score = fp8_index_cpu(
                q,  # [B, L, H, D]
                weights,  # [B, L, H]
                self.k_cache[:bsz, :end_pos],  # [B, C, D]
                torch.ones(bsz, end_pos, dtype=torch.float32),  # unit key scales
            )

        # index_score is summed over heads: [B, L, C] where C == end_pos
        assert index_score.shape == (
            bsz,
            seqlen,
            end_pos,
        ), f"Expected index_score shape {(bsz, seqlen, end_pos)}, got {tuple(index_score.shape)}"

        # Apply mask if provided ([seqlen, end_pos] broadcasts over batch)
        if mask is not None:
            index_score = index_score + mask

        # Top-K selection
        topk_k = min(self.index_topk, end_pos)
        topk_indices = index_score.topk(topk_k, dim=-1)[1]  # [B, L, topk]

        assert topk_indices.shape == (
            bsz,
            seqlen,
            topk_k,
        ), f"Expected topk_indices shape {(bsz, seqlen, topk_k)}, got {tuple(topk_indices.shape)}"

        return topk_indices, index_score


# ===== MLA =====


class MLACPU(nn.Module):
    """
    CPU-compatible version of the MLA (Multi-Head Latent Attention) layer.

    Replaces ColumnParallelLinear / RowParallelLinear with plain Linear
    (world_size == 1, so the all-reduce in ``wo`` is a no-op), drops weight
    dequant (random bf16 weights), and keeps the KV-cache fp8 quant/dequant as a
    precision *simulation* applied identically on every write path.
    """

    def __init__(self, args: ModelArgs, simulate_fp8: bool = True):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.scale_fmt = args.scale_fmt
        # When False, the latent KV cache stores the plain bf16 latent (no fp8
        # precision simulation). Kept True by default to mirror the device's fp8
        # KV cache; disabled by the ttnn port, which stores bf16, so the two match.
        self.simulate_fp8 = simulate_fp8

        # Projections (Column/Row-parallel collapse to plain Linear at world_size=1).
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)

        # softmax scale with YaRN mscale correction (max_seq_len > original_seq_len).
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Nested sparse-attention selector (shares the same args instance).
        self.indexer = IndexerCPU(args)

        # Latent KV cache (the MLA memory win): stores latent kv (512) + k_pe (64).
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16),
            persistent=False,
        )

        logger.info(f"Initialized MLACPU with {self.n_heads} heads, qk_head_dim={self.qk_head_dim}")

    def _simulate_fp8_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Simulate the deployed fp8 KV-cache precision: quantize then dequantize so
        the value stored is what the device would store. Applied on the write
        path in BOTH branches, so decode reads exactly what prefill wrote
        (cache-correctness invariant). See model.py:570-571.
        """
        kv_fp8, kv_scale = act_quant_cpu(kv.contiguous(), block_size=BLOCK_SIZE, scale_fmt=self.scale_fmt)
        kv_deq = (kv_fp8.view(-1, BLOCK_SIZE).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        return kv_deq

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: input features [B, S, dim]
            start_pos: write offset into the cache
            freqs_cis: rotary frequencies for absolute positions [S, rope/2]
            mask: causal mask [S, S] for prefill, or None for decode

        Returns:
            out: [B, S, dim]
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        logger.info(
            f"MLA forward: bsz={bsz}, seqlen={seqlen}, start_pos={start_pos}, mask={'set' if mask is not None else 'None'}"
        )

        # ----- Shared front-end -----
        qr = self.q_norm(self.wq_a(x))  # [B, S, q_lora_rank]
        q = self.wq_b(qr)  # [B, S, H * qk_head_dim]
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # MLA RoPE is interleaved (the indexer's is not).
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        kv = self.wkv_a(x)  # [B, S, kv_lora_rank + rope]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)  # latent kv [B, S, 512]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)  # [B, S, 1, 64]

        # fp8 KV-cache precision simulation (consistent on every write path).
        if self.simulate_fp8:
            kv = self._simulate_fp8_kv(kv)

        # Write current chunk into the latent caches.
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        if mask is not None and start_pos == 0:
            # ----- Prefill / MHA: materialize per-head K and V via wkv_b -----
            # Chunked prefill (start_pos > 0, mask [S, end_pos]) takes the decode/MQA
            # branch below: K/V come from the caches, mask keeps chunk causality.
            q = torch.cat([q_nope, q_pe], dim=-1)  # [B, S, H, qk_head_dim]
            kv_b = self.wkv_b(kv)  # [B, S, H * (nope + v)]
            kv_b = kv_b.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)  # [B, S, H, qk_head_dim]
            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale  # [B, S, H, T]

            # Indexer -> additive {0, -inf} mask, combined with the causal mask.
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)[0]
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            index_mask = index_mask + mask
            scores = scores + index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x_attn = torch.einsum("bsht,bthd->bshd", scores, v)  # [B, S, H, v]
        else:
            # ----- Decode / MQA: absorb wkv_b into Q and output -----
            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)  # [H, 256, c]
            # Absorb the nope half of wkv_b into the query (-> width c).
            q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])  # [B, S, H, c]
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_abs, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale  # [B, S, H, T]

            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)[0]
            index_mask = torch.full((bsz, seqlen, end_pos), float("-inf"), device=x.device).scatter_(
                -1, topk_indices, 0
            )
            # Chunked prefill ([S>1, end_pos] mask): rows with fewer causal keys than
            # topk still get future indices scattered to 0 — re-impose causality like
            # the prefill branch above (mask shape [S, end_pos], rows offset start_pos).
            if mask is not None:
                index_mask = index_mask + mask
            scores = scores + index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x_attn = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])  # [B, S, H, c]
            # Absorb the value half of wkv_b at the very end (-> width v).
            x_attn = torch.einsum("bshc,hdc->bshd", x_attn, wkv_b[:, -self.v_head_dim :])  # [B, S, H, v]

        out = self.wo(x_attn.flatten(2))  # [B, S, dim]
        logger.info(f"MLA output shape: {tuple(out.shape)}")
        return out
