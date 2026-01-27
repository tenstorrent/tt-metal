# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN-accelerated GR00T models."""

from typing import Optional
import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.modules.linear import TTNNLinear


def force_to_ttnn(obj, device):
    """Deep unwrap to bridge Symbiote wrappers and raw C++ hardware tensors."""
    curr = obj
    for _ in range(5):
        if hasattr(curr, "value"):
            curr = curr.value
        elif hasattr(curr, "tensor"):
            curr = curr.tensor
        else:
            break

    if not isinstance(curr, ttnn.Tensor):
        t_torch = obj.to_torch if hasattr(obj, "to_torch") else obj
        if callable(t_torch):
            t_torch = t_torch()
        if not isinstance(t_torch, torch.Tensor):
            t_torch = torch.tensor(t_torch)
        return ttnn.from_torch(t_torch.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    return curr


class TTNNSDPAAttention(nn.Module):
    """Hardware-accelerated Scaled Dot Product Attention (SDPA)."""

    def __init__(self):
        super().__init__()
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )

        # PARITY TWEAK: Using HiFi4 + Dest Accumulation for maximum precision
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # Disabled to prevent small rounding drift
        )

    def forward(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        scaling: Optional[float] = None,
        is_causal: bool = False,
        device=None,
    ) -> ttnn.Tensor:
        """Execute SDPA on Tensix cores with high precision configs."""
        q = ttnn.to_memory_config(ttnn.to_layout(force_to_ttnn(query, device), ttnn.TILE_LAYOUT), self.memory_config)
        k = ttnn.to_memory_config(ttnn.to_layout(force_to_ttnn(key, device), ttnn.TILE_LAYOUT), self.memory_config)
        v = ttnn.to_memory_config(ttnn.to_layout(force_to_ttnn(value, device), ttnn.TILE_LAYOUT), self.memory_config)

        if attention_mask is not None:
            attention_mask = force_to_ttnn(attention_mask, device)

        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scaling,
            program_config=self.program_config,
            attn_mask=attention_mask,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
        )


class TTNNSelfAttention(nn.Module):
    """TTNN implementation of Self-Attention with automatic weight mapping."""

    def __init__(self, torch_layer: Optional[nn.Module] = None):
        super().__init__()
        if torch_layer is None:
            return

        self.num_heads = getattr(torch_layer, "num_heads", 12)
        self.hidden_size = getattr(torch_layer, "hidden_size", 768)
        self.head_dim = self.hidden_size // self.num_heads

        # Scaling is critical for numerical parity
        self.scaling = self.head_dim**-0.5
        self.sdpa = TTNNSDPAAttention()
        self._tt_linears = nn.ModuleDict()

        # Recursive search for weights (supports nested DiT and Backbone)
        weight_map = {"q": ["q_proj", "to_q"], "k": ["k_proj", "to_k"], "v": ["v_proj", "to_v"], "qkv": ["qkv_proj"]}

        found = {}
        for name, m in torch_layer.named_modules():
            for key, aliases in weight_map.items():
                if any(alias == name.split(".")[-1] for alias in aliases) and isinstance(m, nn.Linear):
                    found[key] = m

        if "q" in found and "k" in found:
            self._tt_linears["q"] = TTNNLinear.from_torch(found["q"])
            self._tt_linears["k"] = TTNNLinear.from_torch(found["k"])
            self._tt_linears["v"] = TTNNLinear.from_torch(found["v"])
            self.mode = "split"
        elif "qkv" in found:
            self._tt_linears["qkv"] = TTNNLinear.from_torch(found["qkv"])
            self.mode = "fused"
        else:
            self.mode = "bypass"

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through hardware-mapped attention heads."""
        if self.mode == "bypass":
            return hidden_states

        device = list(self._tt_linears.values())[0].weight.device()
        hw_states = force_to_ttnn(hidden_states, device)

        if hw_states.layout != ttnn.TILE_LAYOUT:
            hw_states = ttnn.to_layout(hw_states, ttnn.TILE_LAYOUT)

        if self.mode == "fused":
            qkv = self._tt_linears["qkv"](hw_states)
            query, key, value = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.num_heads, num_kv_heads=self.num_heads
            )
        else:
            context = kwargs.get("context", hidden_states)
            hw_context = force_to_ttnn(context, device)

            if hw_context.layout != ttnn.TILE_LAYOUT:
                hw_context = ttnn.to_layout(hw_context, ttnn.TILE_LAYOUT)

            q_out = self._tt_linears["q"](hw_states)
            k_out = self._tt_linears["k"](hw_context)
            v_out = self._tt_linears["v"](hw_context)

            query = ttnn.experimental.nlp_create_q_heads(q_out, num_heads=self.num_heads)
            key = ttnn.experimental.nlp_create_k_heads(k_out, num_heads=self.num_heads)
            value = ttnn.experimental.nlp_create_v_heads(v_out, num_heads=self.num_heads)

        # Compute Attention Scores and context
        out = self.sdpa(query, key, value, attention_mask=attention_mask, scaling=self.scaling, device=device)

        return ttnn.experimental.nlp_concat_heads(out)
