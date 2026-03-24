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

import torch

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import create_chunk_masks
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn


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

        # Pre-cache chunk masks for prefill (shared across all calls)
        self.prefill_chunk_size = 64
        self.cached_masks = create_chunk_masks(self.prefill_chunk_size, device)

        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
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

        # After prefill, fuse separate conv states into one for efficient decode
        T = x.shape[1]
        if T == 1 and self.fused_conv_state is None and self.conv_state_q is not None:
            self.fused_conv_state = ttnn.concat([self.conv_state_q, self.conv_state_k, self.conv_state_v], dim=2)
            self.fused_conv_state = ttnn.to_layout(self.fused_conv_state, ttnn.TILE_LAYOUT)

        # Use cached masks for chunk mode with matching chunk_size
        masks = self.cached_masks if (mode == "chunk" and chunk_size == self.prefill_chunk_size) else None

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
            compute_kernel_config=self.compute_kernel_config,
            A_neg_precomputed=self.A_neg,
            fused_conv_weight_taps=self.fused_conv_weight_taps,
            fused_conv_bias_dev=self.fused_conv_bias_dev,
            fused_conv_state=self.fused_conv_state,
            ab_proj_weight=self.ab_proj_weight,
            cached_masks=masks,
            use_inplace_state=self.use_inplace_state,
        )

        self.recurrent_state = new_state
        if new_fused_conv is not None:
            self.fused_conv_state = new_fused_conv
        else:
            self.conv_state_q = new_conv_q
            self.conv_state_k = new_conv_k
            self.conv_state_v = new_conv_v
        return output

    def enable_inplace_state(self):
        """Enable inplace state updates for trace capture."""
        self.use_inplace_state = True

    def init_fused_conv_buffer(self, batch_size):
        """Pre-allocate fused conv state buffer for trace capture."""
        D_total = self.args.linear_q_dim + self.args.linear_k_dim + self.args.linear_v_dim
        state = torch.zeros(batch_size, self.conv_kernel_size - 1, D_total, dtype=torch.bfloat16)
        self.fused_conv_buffer = ttnn.from_torch(
            state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            self._init_recurrent_state(batch_size)
        else:
            self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        self.fused_conv_state = None
