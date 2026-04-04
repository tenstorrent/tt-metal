# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet linear attention for Qwen3-Coder-Next.

DeltaNet is a recurrent linear attention mechanism used in 36 of 48 layers
(every layer where layer_idx % 4 != 3). Unlike softmax attention, it:
- Uses a gating mechanism instead of softmax
- Maintains recurrent state instead of a KV cache
- Includes a short 1D convolution (kernel_dim=4) before attention
- Has O(n) complexity instead of O(n^2) for long sequences

Architecture per DeltaNet layer:
    Input (B, S, H=2048)
    -> Short Conv1D (kernel=4) on queries/keys
    -> Linear projections: Q (16 key heads × 128 dim), K (16 key heads × 128 dim),
                           V (32 value heads × 256 dim)
    -> Gated linear attention (recurrent)
    -> Output projection -> (B, S, H=2048)

Reference: Qwen3NextDeltaNetAttention in HuggingFace transformers >= 4.57.0.dev0
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class GatedDeltaNetAttention(nn.Module):
    """Gated DeltaNet linear attention module.

    This is a PyTorch reference implementation for correctness validation.
    The TTNN implementation will follow once PCC is confirmed.

    Maintains a recurrent state S of shape (batch, num_heads, key_dim, value_dim)
    that gets updated at each timestep instead of growing a KV cache.
    """

    def __init__(self, config: Qwen3CoderNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # DeltaNet-specific dimensions
        self.key_dim = config.linear_key_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 32
        self.value_dim = config.head_dim  # 256 (same as GQA head_dim)

        # Short 1D convolution before attention
        self.conv_kernel_dim = config.linear_conv_kernel_dim  # 4

        # Projections
        # Q and K project to (num_key_heads * key_dim)
        self.q_proj = nn.Linear(self.hidden_size, self.num_key_heads * self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_heads * self.key_dim, bias=False)
        # V projects to (num_value_heads * value_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_value_heads * self.value_dim, bias=False)
        # Output projection
        self.o_proj = nn.Linear(self.num_value_heads * self.value_dim, self.hidden_size, bias=False)

        # Short convolutions for Q and K (causal, applied per-head)
        # Conv1d with groups=num_key_heads for per-head convolution
        self.q_conv = nn.Conv1d(
            in_channels=self.num_key_heads * self.key_dim,
            out_channels=self.num_key_heads * self.key_dim,
            kernel_size=self.conv_kernel_dim,
            groups=self.num_key_heads,
            padding=self.conv_kernel_dim - 1,  # Causal padding
            bias=True,
        )
        self.k_conv = nn.Conv1d(
            in_channels=self.num_key_heads * self.key_dim,
            out_channels=self.num_key_heads * self.key_dim,
            kernel_size=self.conv_kernel_dim,
            groups=self.num_key_heads,
            padding=self.conv_kernel_dim - 1,
            bias=True,
        )

        # Gate projection for gated linear attention
        self.gate_proj = nn.Linear(self.hidden_size, self.num_key_heads, bias=True)

        # Beta (decay) projection
        self.beta_proj = nn.Linear(self.hidden_size, self.num_key_heads, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state_q: Optional[torch.Tensor] = None,
        conv_state_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for Gated DeltaNet attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size).
            recurrent_state: Optional previous recurrent state
                (batch, num_key_heads, key_dim, value_dim_per_head).
                If None, initialized to zeros.
            conv_state_q: Optional convolution state for Q (for incremental decode).
            conv_state_k: Optional convolution state for K.

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_size)
                - new_recurrent_state: Updated recurrent state
                - new_conv_state_q: Updated Q conv state
                - new_conv_state_k: Updated K conv state
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)  # (B, S, num_key_heads * key_dim)
        k = self.k_proj(hidden_states)  # (B, S, num_key_heads * key_dim)
        v = self.v_proj(hidden_states)  # (B, S, num_value_heads * value_dim)

        # Apply short causal convolution to Q and K
        # Transpose for Conv1d: (B, C, S)
        q_conv_input = q.transpose(1, 2)
        k_conv_input = k.transpose(1, 2)

        q = self.q_conv(q_conv_input)[:, :, :seq_len].transpose(1, 2)  # Causal: trim future
        k = self.k_conv(k_conv_input)[:, :, :seq_len].transpose(1, 2)

        # Apply SiLU activation after conv
        q = F.silu(q)
        k = F.silu(k)

        # Compute gate and beta (decay factor)
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # (B, S, num_key_heads)
        beta = torch.sigmoid(self.beta_proj(hidden_states))  # (B, S, num_key_heads)

        # Reshape Q, K for multi-head: (B, S, num_key_heads, key_dim)
        q = q.view(batch_size, seq_len, self.num_key_heads, self.key_dim)
        k = k.view(batch_size, seq_len, self.num_key_heads, self.key_dim)

        # Reshape V: (B, S, num_value_heads, value_dim)
        v = v.view(batch_size, seq_len, self.num_value_heads, self.value_dim)

        # Handle head ratio: num_value_heads / num_key_heads
        # If num_value_heads > num_key_heads, each key head serves multiple value heads
        value_heads_per_key_head = self.num_value_heads // self.num_key_heads  # 32/16 = 2

        # Initialize recurrent state if needed
        # State shape: (B, num_key_heads, key_dim, value_dim * value_heads_per_key_head)
        effective_value_dim = self.value_dim * value_heads_per_key_head
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch_size,
                self.num_key_heads,
                self.key_dim,
                effective_value_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Recurrent linear attention with gating
        # For each timestep t:
        #   S_t = beta_t * S_{t-1} + k_t^T @ v_t  (delta rule update)
        #   o_t = gate_t * (q_t @ S_t)             (gated readout)
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]  # (B, num_key_heads, key_dim)
            k_t = k[:, t]  # (B, num_key_heads, key_dim)
            beta_t = beta[:, t, :, None, None]  # (B, num_key_heads, 1, 1)
            gate_t = gate[:, t, :, None]  # (B, num_key_heads, 1)

            # Group value heads by key head
            v_t = v[:, t]  # (B, num_value_heads, value_dim)
            v_t_grouped = v_t.view(batch_size, self.num_key_heads, value_heads_per_key_head, self.value_dim)
            v_t_flat = v_t_grouped.reshape(
                batch_size, self.num_key_heads, effective_value_dim
            )  # (B, num_key_heads, effective_value_dim)

            # Delta rule: S = beta * S + outer(k, v)
            k_t_col = k_t.unsqueeze(-1)  # (B, num_key_heads, key_dim, 1)
            v_t_row = v_t_flat.unsqueeze(-2)  # (B, num_key_heads, 1, effective_value_dim)
            recurrent_state = beta_t * recurrent_state + k_t_col * v_t_row

            # Gated readout: o = gate * (q @ S)
            q_t_row = q_t.unsqueeze(-2)  # (B, num_key_heads, 1, key_dim)
            o_t = torch.matmul(q_t_row, recurrent_state).squeeze(-2)  # (B, num_key_heads, effective_value_dim)
            o_t = gate_t * o_t  # (B, num_key_heads, effective_value_dim)

            # Reshape back to (B, num_value_heads * value_dim)
            o_t = o_t.view(batch_size, self.num_value_heads * self.value_dim)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # (B, S, num_value_heads * value_dim)
        output = self.o_proj(output)  # (B, S, hidden_size)

        return output, recurrent_state, None, None
