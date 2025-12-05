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
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pre_only = pre_only
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

        # Scaled dot-product attention for main input
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.core_grid,
            q_chunk_size=64,
            k_chunk_size=64,
        )

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
            # Apply dropout if needed (for inference with p=0.0, this is a no-op)
            if self.dropout_prob > 0.0:
                attn_out = ttnn.experimental.dropout(
                    attn_out, probability=self.dropout_prob, scale=1.0 / (1.0 - self.dropout_prob)
                )

        # Handle added context if provided
        if added_input is not None:
            added_B = added_input.shape[1]

            # Added attention projections
            added_q = self.add_q_proj(added_input)
            added_k = self.add_k_proj(added_input)
            added_v = self.add_v_proj(added_input)

            # Apply RMSNorm to added Q and K
            if self.norm_added_q is not None:
                added_q = self.norm_added_q(added_q)
                added_k = self.norm_added_k(added_k)

            # Reshape added tensors
            added_q = ttnn.reshape(added_q, (added_B, added_seq_len, self.num_heads, self.head_dim))
            added_k = ttnn.reshape(added_k, (added_B, added_seq_len, self.num_heads, self.head_dim))
            added_v = ttnn.reshape(added_v, (added_B, added_seq_len, self.num_heads, self.head_dim))

            # Transpose added tensors
            added_q = ttnn.permute(added_q, (0, 2, 1, 3))
            added_k = ttnn.permute(added_k, (0, 2, 1, 3))
            added_v = ttnn.permute(added_v, (0, 2, 1, 3))

            # Cross attention between main query and added key/value
            added_attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,  # Use main query
                added_k,  # Use added key
                added_v,  # Use added value
                is_causal=False,
                scale=self.scale,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config,
            )

            # Transpose and reshape added attention output
            added_attn_out = ttnn.permute(added_attn_out, (0, 2, 1, 3))
            added_attn_out = ttnn.reshape(added_attn_out, (1, added_B, added_seq_len, self.inner_dim))

            # Project added attention output back to added dimension
            added_attn_out = self.to_add_out(added_attn_out)

            return attn_out, added_attn_out

        return attn_out

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load main attention weights
        self.to_q.load_torch_state_dict(substate(state_dict, "to_q"))
        self.to_k.load_torch_state_dict(substate(state_dict, "to_k"))
        self.to_v.load_torch_state_dict(substate(state_dict, "to_v"))

        # Load added context weights
        self.add_q_proj.load_torch_state_dict(substate(state_dict, "add_q_proj"))
        self.add_k_proj.load_torch_state_dict(substate(state_dict, "add_k_proj"))
        self.add_v_proj.load_torch_state_dict(substate(state_dict, "add_v_proj"))
        self.to_add_out.load_torch_state_dict(substate(state_dict, "to_add_out"))

        # Load output projection weights
        if not self.pre_only:
            self.to_out.load_torch_state_dict(substate(state_dict, "to_out.0"))

        # Load normalization weights
        if self.norm_q is not None:
            self.norm_q.load_torch_state_dict(substate(state_dict, "norm_q"))
            self.norm_k.load_torch_state_dict(substate(state_dict, "norm_k"))
            self.norm_added_q.load_torch_state_dict(substate(state_dict, "norm_added_q"))
            self.norm_added_k.load_torch_state_dict(substate(state_dict, "norm_added_k"))
