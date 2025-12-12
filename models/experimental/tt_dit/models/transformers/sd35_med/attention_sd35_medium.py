# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Self-Attention Implementation
Modified to support separate Q/K/V projections and added context.
"""

import ttnn
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.normalization import RMSNorm
from models.experimental.tt_dit.utils.substate import substate


class SD35MediumSelfAttention:
    """Self-attention for SD3.5 Medium with added context support."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        pre_only=False,
        qk_norm="rms",
        eps=1e-6,
        mesh_device=None,
        added_proj_dim=None,
        context_pre_only=False,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pre_only = pre_only
        self.context_pre_only = context_pre_only
        self.inner_dim = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.mesh_device = mesh_device
        self.added_proj_dim = added_proj_dim or dim

        # Separate Q, K, V projections
        self.to_q = Linear(dim, dim, bias=qkv_bias, mesh_device=mesh_device)
        self.to_k = Linear(dim, dim, bias=qkv_bias, mesh_device=mesh_device)
        self.to_v = Linear(dim, dim, bias=qkv_bias, mesh_device=mesh_device)

        # Additional projections for added context
        self.add_q_proj = Linear(self.added_proj_dim, dim, bias=qkv_bias, mesh_device=mesh_device)
        self.add_k_proj = Linear(self.added_proj_dim, dim, bias=qkv_bias, mesh_device=mesh_device)
        self.add_v_proj = Linear(self.added_proj_dim, dim, bias=qkv_bias, mesh_device=mesh_device)
        # to_add_out only needed when context_pre_only=False
        if not context_pre_only:
            self.to_add_out = Linear(dim, self.added_proj_dim, mesh_device=mesh_device)

        # Output projection - dropout handled in forward pass for inference
        if not pre_only:
            self.to_out = Linear(self.inner_dim, dim, mesh_device=mesh_device)
            self.dropout_prob = 0.0  # Store dropout probability

        # QK normalization for main attention
        if qk_norm == "rms":
            self.norm_q = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
            self.norm_k = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
        else:
            self.norm_q = None
            self.norm_k = None

        # QK normalization for added attention
        if qk_norm == "rms":
            self.norm_added_q = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
            self.norm_added_k = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # Compute config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.core_grid = mesh_device.compute_with_storage_grid_size()

    def __call__(self, x, seq_len, added_input=None, added_seq_len=None):
        """
        Forward pass.
        Args:
            x: [1, B, seq_len, dim] - main input
            seq_len: sequence length for main input
            added_input: [1, B, added_seq_len, added_proj_dim] - optional added context
            added_seq_len: sequence length for added input
        Returns:
            [1, B, seq_len, dim] or tuple if added_input provided
        """
        B = x.shape[1]

        # Main attention projections
        q = self.to_q(x)  # [1, B, seq_len, dim]
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to [B, seq_len, num_heads, head_dim] BEFORE applying RMSNorm
        q = ttnn.reshape(q, (B, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (B, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (B, seq_len, self.num_heads, self.head_dim))

        # Apply RMSNorm to Q and K AFTER reshaping (now head_dim=64 matches norm)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Transpose to [B, num_heads, seq_len, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # SDPA program config
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.core_grid,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        # Handle added context if provided - SD3 uses JOINT attention (concatenate, attend, split)
        if added_input is not None:
            added_B = added_input.shape[1]

            # Added attention projections
            added_q = self.add_q_proj(added_input)
            added_k = self.add_k_proj(added_input)
            added_v = self.add_v_proj(added_input)

            # Reshape added tensors to [B, added_seq_len, num_heads, head_dim] BEFORE applying RMSNorm
            added_q = ttnn.reshape(added_q, (added_B, added_seq_len, self.num_heads, self.head_dim))
            added_k = ttnn.reshape(added_k, (added_B, added_seq_len, self.num_heads, self.head_dim))
            added_v = ttnn.reshape(added_v, (added_B, added_seq_len, self.num_heads, self.head_dim))

            # Apply RMSNorm to added Q and K AFTER reshaping (now head_dim=64 matches norm)
            if self.norm_added_q is not None:
                added_q = self.norm_added_q(added_q)
                added_k = self.norm_added_k(added_k)

            # Transpose added tensors to [B, num_heads, added_seq_len, head_dim]
            added_q = ttnn.permute(added_q, (0, 2, 1, 3))
            added_k = ttnn.permute(added_k, (0, 2, 1, 3))
            added_v = ttnn.permute(added_v, (0, 2, 1, 3))

            # JOINT ATTENTION: Concatenate x and context along sequence dimension
            # q: [B, num_heads, seq_len, head_dim], added_q: [B, num_heads, added_seq_len, head_dim]
            # Result: [B, num_heads, seq_len + added_seq_len, head_dim]
            q_full = ttnn.concat([q, added_q], dim=2)
            k_full = ttnn.concat([k, added_k], dim=2)
            v_full = ttnn.concat([v, added_v], dim=2)

            # Run unified self-attention on concatenated sequence
            attn_out_full = ttnn.transformer.scaled_dot_product_attention(
                q_full,
                k_full,
                v_full,
                is_causal=False,
                scale=self.scale,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            # Output shape: [B, num_heads, seq_len + added_seq_len, head_dim]

            # Split output back into x and context parts
            # attn_out: [B, num_heads, seq_len, head_dim]
            # added_attn_out: [B, num_heads, added_seq_len, head_dim]
            attn_out = attn_out_full[:, :, :seq_len, :]
            added_attn_out = attn_out_full[:, :, seq_len:, :]

            # Transpose back: [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads, head_dim]
            attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
            added_attn_out = ttnn.permute(added_attn_out, (0, 2, 1, 3))

            # Reshape to [1, B, seq_len, inner_dim]
            attn_out = ttnn.reshape(attn_out, (1, B, seq_len, self.inner_dim))
            added_attn_out = ttnn.reshape(added_attn_out, (1, added_B, added_seq_len, self.inner_dim))

            # Output projections
            if not self.pre_only:
                attn_out = self.to_out(attn_out)
                if self.dropout_prob > 0.0:
                    attn_out = ttnn.experimental.dropout(
                        attn_out, probability=self.dropout_prob, scale=1.0 / (1.0 - self.dropout_prob)
                    )

            # Project added attention output back to added dimension (skip for final block)
            if not self.context_pre_only:
                added_attn_out = self.to_add_out(added_attn_out)

            return attn_out, added_attn_out

        # No added input - just self-attention on x
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Transpose back: [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads, head_dim]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))

        # Reshape to [1, B, seq_len, inner_dim]
        attn_out = ttnn.reshape(attn_out, (1, B, seq_len, self.inner_dim))

        # Output projection
        if not self.pre_only:
            attn_out = self.to_out(attn_out)
            if self.dropout_prob > 0.0:
                attn_out = ttnn.experimental.dropout(
                    attn_out, probability=self.dropout_prob, scale=1.0 / (1.0 - self.dropout_prob)
                )

        return attn_out

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load main attention weights
        self.to_q.load_torch_state_dict(substate(state_dict, "to_q"))
        self.to_k.load_torch_state_dict(substate(state_dict, "to_k"))
        self.to_v.load_torch_state_dict(substate(state_dict, "to_v"))

        # Load added context weights (only if present - attn2 doesn't have these)
        add_q_state = substate(state_dict, "add_q_proj")
        if add_q_state:
            self.add_q_proj.load_torch_state_dict(add_q_state)
            self.add_k_proj.load_torch_state_dict(substate(state_dict, "add_k_proj"))
            self.add_v_proj.load_torch_state_dict(substate(state_dict, "add_v_proj"))
            # to_add_out only exists in early/middle blocks, not final block
            if not self.context_pre_only:
                to_add_out_state = substate(state_dict, "to_add_out")
                if to_add_out_state:
                    self.to_add_out.load_torch_state_dict(to_add_out_state)

        # Load output projection weights
        if not self.pre_only:
            self.to_out.load_torch_state_dict(substate(state_dict, "to_out.0"))

        # Load normalization weights
        if self.norm_q is not None:
            self.norm_q.load_torch_state_dict(substate(state_dict, "norm_q"))
            self.norm_k.load_torch_state_dict(substate(state_dict, "norm_k"))
            # Only load added norm weights if present (attn2 doesn't have these)
            norm_added_q_state = substate(state_dict, "norm_added_q")
            if norm_added_q_state:
                self.norm_added_q.load_torch_state_dict(norm_added_q_state)
                self.norm_added_k.load_torch_state_dict(substate(state_dict, "norm_added_k"))
