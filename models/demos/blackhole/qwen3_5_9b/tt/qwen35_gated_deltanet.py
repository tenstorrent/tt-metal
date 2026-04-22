# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet wrapper for Qwen3.5-9B linear attention layers.

Wraps the experimental `gated_deltanet_forward_ttnn()` into a module
that manages weight tensors, recurrent state, and conv state.
"""
import os
import sys

# The experimental gated_deltanet module uses `from tt.ttnn_delta_rule_ops import ...`
# which requires its parent directory on sys.path so `tt` resolves as a package.
_EXPERIMENTAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "experimental", "gated_attention_gated_deltanet"
)
_EXPERIMENTAL_DIR = os.path.abspath(_EXPERIMENTAL_DIR)
if _EXPERIMENTAL_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTAL_DIR)

import math

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import create_chunk_masks
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import (
    gated_deltanet_forward_ttnn,
    rms_norm_gated_ttnn,
)


class Qwen35GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
        self.args = args
        self.device = device
        self.layer_num = layer_num
        self.num_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.norm_eps = args.norm_eps

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        prefix = f"layers.{layer_num}.linear_attn"

        def load_weight_2d(name):
            """Load 2D weight, transpose to [in, out] for ttnn.linear."""
            t = state_dict[f"{prefix}.{name}"].T.contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
            )

        def load_conv_weight(name):
            """Load conv1d weight — stays on HOST (not device), ROW_MAJOR layout."""
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

        def load_1d(name):
            """Load 1D param — must use TILE_LAYOUT on device like all other tensors."""
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
            )

        # Fused QKV projection: one matmul instead of three
        qkv_key = f"{prefix}.qkv_proj.weight"
        if qkv_key in state_dict:
            t = state_dict[qkv_key].T.contiguous()  # [4096, 8192]
            self.qkv_proj_weight = ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.qkv_proj.weight" if weight_cache_path else None,
            )
        else:
            self.qkv_proj_weight = None
        # Keep separate weights as fallback and for conv weight tap dimensions
        self.q_proj_weight = load_weight_2d("q_proj.weight")
        self.k_proj_weight = load_weight_2d("k_proj.weight")
        self.v_proj_weight = load_weight_2d("v_proj.weight")
        self.a_proj_weight = load_weight_2d("in_proj_a.weight")
        self.b_proj_weight = load_weight_2d("in_proj_b.weight")
        self.g_proj_weight = load_weight_2d("in_proj_z.weight")
        self.o_proj_weight = load_weight_2d("out_proj.weight")

        self.q_conv_weight = load_conv_weight("q_conv.weight")
        self.k_conv_weight = load_conv_weight("k_conv.weight")
        self.v_conv_weight = load_conv_weight("v_conv.weight")

        def load_conv_bias_or_none(name):
            full_key = f"{prefix}.{name}"
            if full_key in state_dict:
                t = state_dict[full_key]
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            return None

        self.q_conv_bias = load_conv_bias_or_none("q_conv.bias")
        self.k_conv_bias = load_conv_bias_or_none("k_conv.bias")
        self.v_conv_bias = load_conv_bias_or_none("v_conv.bias")

        self.A_log = load_1d("A_log")
        self.dt_bias = load_1d("dt_bias")
        # DeltaNet output norm uses STANDARD RMSNorm (raw weights ~0.88),
        # NOT zero-centered like the decoder/attention norms (raw weights ~0.03).
        self.o_norm_weight = load_1d("norm.weight")

        # Precompute A_neg = -exp(A_log) once (constant per layer, saves 2 ops per decode step)
        self.A_neg = ttnn.neg(ttnn.exp(self.A_log))

        # Precompute conv weight taps and bias on device to avoid CPU round-trips during decode
        self.q_weight_taps = self._precompute_weight_taps(self.q_conv_weight)
        self.k_weight_taps = self._precompute_weight_taps(self.k_conv_weight)
        self.v_weight_taps = self._precompute_weight_taps(self.v_conv_weight)
        self.q_bias_dev = self._precompute_bias_dev(self.q_conv_bias)
        self.k_bias_dev = self._precompute_bias_dev(self.k_conv_bias)
        self.v_bias_dev = self._precompute_bias_dev(self.v_conv_bias)

        # Precompute fused QKV conv weight taps [1, 1, D_total] for fused conv decode
        self.fused_conv_weight_taps = self._precompute_fused_weight_taps()
        self.fused_conv_bias_dev = self._precompute_fused_bias_dev()

        # Fused a+b projection weight: [4096, 64] — saves 1 matmul per decode step
        self.ab_proj_weight = self._precompute_fused_ab_weight()

        # Mega-fused weight: QKV + a + b + g in one [4096, 12352] matmul
        # Eliminates 2 separate matmuls (g_proj, ab_proj) per decode step
        self.mega_fused_weight = self._precompute_mega_fused_weight()
        if self.mega_fused_weight is not None:
            self.mega_qkv_dim = self.args.linear_q_dim + self.args.linear_k_dim + self.args.linear_v_dim
            self.mega_a_dim = self.num_v_heads
            self.mega_b_dim = self.num_v_heads
            self.mega_g_dim = self.num_v_heads * self.head_v_dim
        else:
            self.mega_qkv_dim = None
            self.mega_a_dim = None
            self.mega_b_dim = None
            self.mega_g_dim = None

        # Pre-cache chunk masks for prefill (shared across all calls)
        self.prefill_chunk_size = 64
        self.cached_masks = create_chunk_masks(self.prefill_chunk_size, device)

        # The Neumann approximation overflows at chunk_size=256, but we now use
        # exact forward substitution (LAPACK triangular solve) which is numerically
        # stable at any chunk size. chunk_size=128 balances device compute (128x128
        # matmuls) vs sub-chunk count (8 per 1024-token outer chunk) vs LAPACK
        # solve cost ([256, 128, 128] = 16MB transfer).
        # NOTE: chunk_size=64 is ~39% faster per-layer but compounds PCC errors
        # across 32 layers (PCC drops from 0.999 single-layer to 0.87 full-model).
        self.long_prefill_chunk_size = 128
        self.cached_masks_long = create_chunk_masks(self.long_prefill_chunk_size, device)

        # Prefill kernel constants (precomputed once, reused across all prefill calls)
        self._use_prefill_kernel = not os.environ.get("GDN_DISABLE_PREFILL_KERNEL", "")
        if self._use_prefill_kernel:
            self._precompute_prefill_kernel_constants(device)

        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
        self.split_conv_state = None
        # Trace capture support
        self.use_inplace_state = False

    def _precompute_weight_taps(self, conv_weight):
        """Pre-slice conv weight [D, 1, K] into K device tensors [1, 1, D] for FIR decode."""

        weight_torch = ttnn.to_torch(conv_weight)
        D = weight_torch.shape[0]
        taps = []
        for k in range(self.conv_kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device))
        return taps

    def _precompute_bias_dev(self, conv_bias):
        """Pre-convert conv bias to [1, 1, D] device tensor."""
        if conv_bias is None:
            return None

        bias_torch = ttnn.to_torch(conv_bias)
        D = bias_torch.numel()
        bias_reshaped = bias_torch.reshape(1, 1, D).contiguous()
        return ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _precompute_fused_weight_taps(self):
        """Pre-concatenate Q+K+V conv weight taps into fused taps [1, 1, D_total]."""
        import torch

        q_w = ttnn.to_torch(self.q_conv_weight)  # [D_q, 1, K]
        k_w = ttnn.to_torch(self.k_conv_weight)  # [D_k, 1, K]
        v_w = ttnn.to_torch(self.v_conv_weight)  # [D_v, 1, K]
        # Concatenate along D dimension: [D_total, 1, K]
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        D_total = fused_w.shape[0]
        taps = []
        for k_idx in range(self.conv_kernel_size):
            w_k = fused_w[:, 0, k_idx].reshape(1, 1, D_total).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device))
        return taps

    def _precompute_fused_bias_dev(self):
        """Pre-concatenate Q+K+V conv biases into fused bias [1, 1, D_total]."""
        import torch

        parts = []
        for bias in [self.q_conv_bias, self.k_conv_bias, self.v_conv_bias]:
            if bias is not None:
                parts.append(ttnn.to_torch(bias))
            else:
                return None  # If any is None, skip fused bias
        fused = torch.cat(parts, dim=0)
        D_total = fused.numel()
        fused_reshaped = fused.reshape(1, 1, D_total).contiguous()
        return ttnn.from_torch(fused_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _precompute_fused_ab_weight(self):
        """Pre-concatenate a_proj + b_proj weights into [4096, 64] for fused matmul."""
        import torch

        a_w = ttnn.to_torch(self.a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(self.b_proj_weight)  # [4096, 32]
        fused = torch.cat([a_w, b_w], dim=1).contiguous()  # [4096, 64]
        return ttnn.from_torch(fused, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _precompute_mega_fused_weight(self):
        """Fuse QKV + a + b + g projections into one [4096, D_total] weight.

        Saves 2 matmul kernel launches per decode step (QKV=1, ab=1, g=1 -> mega=1).
        Output split: [qkv_dim | a_dim | b_dim | g_dim]
        """
        import torch

        if self.qkv_proj_weight is None:
            return None
        qkv_w = ttnn.to_torch(self.qkv_proj_weight)  # [4096, 8192]
        a_w = ttnn.to_torch(self.a_proj_weight)  # [4096, 32]
        b_w = ttnn.to_torch(self.b_proj_weight)  # [4096, 32]
        g_w = ttnn.to_torch(self.g_proj_weight)  # [4096, 4096]
        fused = torch.cat([qkv_w, a_w, b_w, g_w], dim=1).contiguous()
        return ttnn.from_torch(fused, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _init_recurrent_state(self, batch_size):
        """Initialize recurrent state to zeros [B, num_v_heads, head_k_dim, head_v_dim]."""
        state = torch.zeros(
            batch_size,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=torch.bfloat16,
        )
        self.recurrent_state = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward(self, x, mode="recurrent", chunk_size=None):
        if chunk_size is None:
            chunk_size = self.prefill_chunk_size if mode == "chunk" else 64

        if self.recurrent_state is None:
            shape = x.shape
            batch_size = shape[0] if len(shape) == 3 else 1
            self._init_recurrent_state(batch_size)

        # Use kernel-based prefill when available (replaces chunked delta rule)
        T = x.shape[1]
        if mode == "chunk" and T > 1 and self._use_prefill_kernel:
            return self.forward_prefill_kernel(x)

        # After prefill, fuse separate conv states into one for efficient decode
        if T == 1 and self.fused_conv_state is None and self.conv_state_q is not None:
            self.fused_conv_state = ttnn.concat([self.conv_state_q, self.conv_state_k, self.conv_state_v], dim=2)
            self.fused_conv_state = ttnn.to_layout(self.fused_conv_state, ttnn.TILE_LAYOUT)
            self._split_fused_conv_state()

        # Use cached masks for chunk mode with matching chunk_size
        if mode == "chunk" and chunk_size == self.prefill_chunk_size:
            masks = self.cached_masks
        elif mode == "chunk" and chunk_size == self.long_prefill_chunk_size:
            masks = self.cached_masks_long
        else:
            masks = None

        output, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv = gated_deltanet_forward_ttnn(
            hidden_states=x,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            a_proj_weight=self.a_proj_weight,
            b_proj_weight=self.b_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_conv_weight=self.q_conv_weight,
            k_conv_weight=self.k_conv_weight,
            v_conv_weight=self.v_conv_weight,
            q_conv_bias=self.q_conv_bias,
            k_conv_bias=self.k_conv_bias,
            v_conv_bias=self.v_conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            o_norm_weight=self.o_norm_weight,
            g_proj_weight=self.g_proj_weight,
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            use_gate=True,
            norm_eps=self.norm_eps,
            device=self.device,
            recurrent_state=self.recurrent_state,
            conv_state_q=self.conv_state_q,
            conv_state_k=self.conv_state_k,
            conv_state_v=self.conv_state_v,
            mode=mode,
            chunk_size=chunk_size,
            q_weight_taps=self.q_weight_taps,
            k_weight_taps=self.k_weight_taps,
            v_weight_taps=self.v_weight_taps,
            q_bias_dev=self.q_bias_dev,
            k_bias_dev=self.k_bias_dev,
            v_bias_dev=self.v_bias_dev,
            qkv_proj_weight=self.qkv_proj_weight,
            q_dim=self.args.linear_q_dim,
            k_dim=self.args.linear_k_dim,
            compute_kernel_config=self.compute_kernel_config_decode
            if mode == "recurrent"
            else self.compute_kernel_config,
            A_neg_precomputed=self.A_neg,
            fused_conv_weight_taps=self.fused_conv_weight_taps,
            fused_conv_bias_dev=self.fused_conv_bias_dev,
            fused_conv_state=self.fused_conv_state,
            fused_conv_state_split=getattr(self, "split_conv_state", None),
            ab_proj_weight=self.ab_proj_weight,
            mega_fused_weight=self.mega_fused_weight,
            mega_qkv_dim=self.mega_qkv_dim,
            mega_a_dim=self.mega_a_dim,
            mega_b_dim=self.mega_b_dim,
            mega_g_dim=self.mega_g_dim,
            cached_masks=masks,
            use_inplace_state=self.use_inplace_state,
        )

        self.recurrent_state = new_state
        if isinstance(new_fused_conv, list):
            self.split_conv_state = new_fused_conv
        elif new_fused_conv is not None:
            self.fused_conv_state = new_fused_conv
        else:
            self.conv_state_q = new_conv_q
            self.conv_state_k = new_conv_k
            self.conv_state_v = new_conv_v
        return output

    def _precompute_prefill_kernel_constants(self, device):
        """Pre-compute device tensors needed by the GDN prefill kernel."""
        Nv = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        # neg_exp_A: [1, 1, Nv] — already have self.A_neg but need correct shape
        neg_exp_A_torch = ttnn.to_torch(self.A_neg).float().reshape(1, 1, Nv)
        self.kernel_neg_exp_A = ttnn.from_torch(
            neg_exp_A_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # dt_bias: [1, 1, Nv]
        dt_bias_torch = ttnn.to_torch(self.dt_bias).float().reshape(1, 1, Nv)
        self.kernel_dt_bias = ttnn.from_torch(
            dt_bias_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # norm_w: [1, 1, Dv] — RMS norm weight (reader loads Vt tiles, not used by compute)
        norm_w_torch = ttnn.to_torch(self.o_norm_weight).float().reshape(1, 1, Dv)
        self.kernel_norm_w = ttnn.from_torch(
            norm_w_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # scale_tt: Dk^(-0.5) as [1, 1, 1] scalar
        self.kernel_scale_tt = ttnn.from_torch(
            torch.tensor([[[Dk**-0.5]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # rms_scale_tt: sqrt(Dv) as [1, 1, 1] — loaded by reader, not used by compute
        self.kernel_rms_scale_tt = ttnn.from_torch(
            torch.tensor([[[math.sqrt(Dv)]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # rms_eps_tt: Dv * eps as [1, 1, 1] — loaded by reader, not used by compute
        self.kernel_rms_eps_tt = ttnn.from_torch(
            torch.tensor([[[Dv * self.norm_eps]]], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_prefill_kernel(self, x):
        """Prefill using on-device GDN recurrence kernel.

        Replaces the chunked delta rule with a single kernel dispatch that
        processes all tokens sequentially on-device. State stays in L1 across
        tokens — no CPU round-trips, no chunking approximation.

        Input:  x [B, T, hidden_size]
        Output: [B, T, hidden_size]
        """
        B = x.shape[0]
        T = x.shape[1]
        Nv = self.num_v_heads
        Nk = self.num_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim
        q_dim = self.args.linear_q_dim
        k_dim = self.args.linear_k_dim
        v_dim = self.args.linear_v_dim
        qkv_dim = q_dim + k_dim + v_dim
        num_pairs = B * Nv
        K = self.conv_kernel_size

        # Match old path: L1 for short sequences, DRAM for long.
        # Kernel inputs must be in DRAM (NOC reads), but projections/conv use mc.
        mc = ttnn.DRAM_MEMORY_CONFIG if T > 512 else None
        ckc = self.compute_kernel_config

        # ---- 1. Projections (single mega-fused matmul for QKV+a+b+g) ----
        mega_out = ttnn.linear(x, self.mega_fused_weight, memory_config=mc, compute_kernel_config=ckc)
        # Split: QKV | a | b | g
        qkv = mega_out[:, :, : self.mega_qkv_dim]
        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
        a_fused = mega_out[:, :, self.mega_qkv_dim : self.mega_qkv_dim + self.mega_a_dim]
        a_fused = ttnn.to_layout(a_fused, ttnn.TILE_LAYOUT)
        b_fused = mega_out[
            :, :, self.mega_qkv_dim + self.mega_a_dim : self.mega_qkv_dim + self.mega_a_dim + self.mega_b_dim
        ]
        b_fused = ttnn.to_layout(b_fused, ttnn.TILE_LAYOUT)
        gate_raw = mega_out[:, :, self.mega_qkv_dim + self.mega_a_dim + self.mega_b_dim :]
        gate_raw = ttnn.to_layout(gate_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(mega_out)

        # ---- 2. Fused causal conv1d on QKV ----
        if self.fused_conv_state is not None:
            x_padded = ttnn.concat([self.fused_conv_state, qkv], dim=1, memory_config=mc)
        elif self.conv_state_q is not None:
            fused_state = ttnn.concat([self.conv_state_q, self.conv_state_k, self.conv_state_v], dim=2)
            fused_state = ttnn.to_layout(fused_state, ttnn.TILE_LAYOUT)
            x_padded = ttnn.concat([fused_state, qkv], dim=1, memory_config=mc)
        else:
            pad = ttnn.zeros(
                [B, K - 1, qkv_dim],
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
            )
            x_padded = ttnn.concat([pad, qkv], dim=1, memory_config=mc)

        # Save new conv state: last K-1 tokens from padded (pre-conv, pre-silu)
        total_len = (K - 1) + T
        start = total_len - (K - 1)
        new_fused_conv = x_padded[:, start:, :]
        new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)

        # FIR conv1d: sum of shifted slices weighted by conv taps
        conv_out = None
        for j in range(K):
            x_slice = x_padded[:, j : j + T]
            x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
            term = ttnn.multiply(x_slice, self.fused_conv_weight_taps[j], memory_config=mc)
            conv_out = term if conv_out is None else ttnn.add(conv_out, term, memory_config=mc)
        ttnn.deallocate(x_padded)

        if self.fused_conv_bias_dev is not None:
            conv_out = ttnn.add(conv_out, self.fused_conv_bias_dev, memory_config=mc)

        conv_out = ttnn.silu(conv_out, memory_config=mc)
        # conv_out: [B, T, qkv_dim] — post-conv+silu

        # ---- 3. Prepare state for kernel ----
        # Reshape: [B, Nv, Dk, Dv] → [B*Nv, Dk, Dv] for kernel tile addressing
        # State is already TILE_LAYOUT + DRAM from _init_recurrent_state or prior kernel write.
        state_3d = ttnn.reshape(self.recurrent_state, [num_pairs, Dk, Dv])

        # Allocate flat output buffer on device (no host transfer)
        prefill_output = ttnn.zeros(
            [num_pairs * T, 1, Dv],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Ensure kernel inputs are in DRAM for NOC reads (may be in L1 when mc=None)
        if mc != ttnn.DRAM_MEMORY_CONFIG:
            conv_out = ttnn.to_memory_config(conv_out, ttnn.DRAM_MEMORY_CONFIG)
            a_fused = ttnn.to_memory_config(a_fused, ttnn.DRAM_MEMORY_CONFIG)
            b_fused = ttnn.to_memory_config(b_fused, ttnn.DRAM_MEMORY_CONFIG)

        # ---- 4. Call prefill kernel ----
        gdn_prefill_fused(
            conv_out,
            a_fused,
            b_fused,
            self.kernel_neg_exp_A,
            self.kernel_dt_bias,
            self.kernel_norm_w,
            self.kernel_scale_tt,
            self.kernel_rms_scale_tt,
            self.kernel_rms_eps_tt,
            state_3d,
            prefill_output,
            num_pairs=num_pairs,
            num_tokens=T,
            Nv_TP=Nv,
            Nk_TP=Nk,
            repeat_factor=Nv // Nk,
            key_dim_tp=q_dim,
        )
        ttnn.deallocate(conv_out)
        ttnn.deallocate(a_fused)
        ttnn.deallocate(b_fused)

        # ---- 5. Update recurrent state ----
        self.recurrent_state = ttnn.reshape(state_3d, [B, Nv, Dk, Dv])

        # ---- 6. Reshape kernel output ----
        # Flat [num_pairs * T, 1, Dv] → [B, Nv, T, Dv] → transpose → [B, T, Nv, Dv]
        out_4d = ttnn.reshape(prefill_output, [B, Nv, T, Dv])
        ttnn.deallocate(prefill_output)
        out_4d = ttnn.transpose(out_4d, 1, 2)  # [B, T, Nv, Dv]

        # ---- 7. Gated RMS norm with z ----
        gate = ttnn.reshape(gate_raw, [B, T, Nv, Dv])
        o = rms_norm_gated_ttnn(out_4d, gate, self.o_norm_weight, eps=self.norm_eps, memory_config=mc)
        ttnn.deallocate(out_4d)
        ttnn.deallocate(gate_raw)

        # ---- 8. Reshape and output projection ----
        o = ttnn.clip(o, min=-1e4, max=1e4)
        o = ttnn.reshape(o, [B, T, Nv * Dv])
        if mc is not None:
            o = ttnn.to_memory_config(o, mc)
        o = ttnn.linear(o, self.o_proj_weight, memory_config=mc, compute_kernel_config=ckc)

        # ---- 9. Update conv state for next chunk / decode ----
        # new_fused_conv is already TILE_LAYOUT from line 497.
        # Don't split here — splitting is deferred to first decode call
        # (either via prefill_paged post-processing or lazy on first T=1 forward).
        self.fused_conv_state = new_fused_conv
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

        return o

    def set_external_state(self, recurrent_state, conv_state):
        """Point layer at externally-allocated state buffers.
        Sets use_inplace_state=True so all forward passes write state inplace (preserving buffer addresses).
        Does NOT create split_conv_state — that happens after prefill when there is real data to split.
        """
        expected_rec = [1, self.num_v_heads, self.head_k_dim, self.head_v_dim]
        assert (
            list(recurrent_state.shape) == expected_rec
        ), f"recurrent_state shape mismatch: {list(recurrent_state.shape)} != {expected_rec}"
        assert (
            conv_state.shape[1] == self.conv_kernel_size - 1
        ), f"conv_state dim 1 mismatch: {conv_state.shape[1]} != {self.conv_kernel_size - 1}"
        self.recurrent_state = recurrent_state
        self.fused_conv_state = conv_state
        self.use_inplace_state = True

    def _split_fused_conv_state(self):
        """Convert fused conv state [B, 3, D_total] into list of 3 [B, 1, D_total] tensors."""
        if self.fused_conv_state is None:
            return
        self.split_conv_state = []
        for k in range(self.conv_kernel_size - 1):
            s_k = self.fused_conv_state[:, k : k + 1, :]
            s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
            buf = ttnn.clone(s_k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s_k)
            self.split_conv_state.append(buf)

    def _restore_split_conv_from_fused(self):
        """Copy fused_conv_state slices into existing split_conv_state buffers.
        Preserves device addresses (critical for trace replay).
        Use instead of _split_fused_conv_state() when split buffers already exist.
        """
        if self.split_conv_state is None:
            return
        for k in range(self.conv_kernel_size - 1):
            s_k = self.fused_conv_state[:, k : k + 1, :]
            s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
            ttnn.copy(s_k, self.split_conv_state[k])
            ttnn.deallocate(s_k)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            self._init_recurrent_state(batch_size)
        else:
            self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        self.fused_conv_state = None
        self.split_conv_state = None
