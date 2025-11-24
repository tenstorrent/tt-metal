# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Self-Attention Implementation
Single-input self-attention matching MM-DiT reference.
"""

import ttnn
from ...layers.linear import Linear
from ...layers.normalization import RMSNorm
from ...utils.substate import substate


class SD35MediumSelfAttention:
    """Single-input self-attention for SD3.5 Medium."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        pre_only=False,
        qk_norm="rms",
        eps=1e-6,
        mesh_device=None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pre_only = pre_only
        self.inner_dim = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.mesh_device = mesh_device

        # Fused QKV projection
        self.qkv = Linear(dim, 3 * self.inner_dim, bias=qkv_bias, mesh_device=mesh_device)

        # Output projection
        if not pre_only:
            self.proj = Linear(self.inner_dim, dim, mesh_device=mesh_device)

        # QK normalization
        if qk_norm == "rms":
            self.ln_q = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
            self.ln_k = RMSNorm(
                embedding_dim=self.head_dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )
        else:
            self.ln_q = None
            self.ln_k = None

        # Compute config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.core_grid = mesh_device.compute_with_storage_grid_size()

    def __call__(self, x, seq_len):
        """
        Forward pass.
        Args:
            x: [1, B, seq_len, dim]
            seq_len: sequence length
        Returns:
            [1, B, seq_len, dim]
        """
        B = x.shape[1]

        # Fused QKV projection
        qkv = self.qkv(x)  # [1, B, seq_len, 3*inner_dim]

        # Reshape and split: [1, B, seq_len, 3, num_heads, head_dim]
        qkv = ttnn.reshape(qkv, (1, B, seq_len, 3, self.num_heads, self.head_dim))

        # Split Q, K, V
        q = qkv[:, :, :, 0, :, :]  # [1, B, seq_len, num_heads, head_dim]
        k = qkv[:, :, :, 1, :, :]
        v = qkv[:, :, :, 2, :, :]

        # Apply RMSNorm to Q and K
        if self.ln_q is not None:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # Reshape to [B, seq_len, num_heads, head_dim]
        q = ttnn.reshape(q, (B, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (B, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (B, seq_len, self.num_heads, self.head_dim))

        # Transpose to [B, num_heads, seq_len, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Scaled dot-product attention
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
            attn_out = self.proj(attn_out)

        return attn_out

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.qkv.load_torch_state_dict(substate(state_dict, "qkv"))

        if not self.pre_only:
            self.proj.load_torch_state_dict(substate(state_dict, "proj"))

        if self.ln_q is not None:
            self.ln_q.load_torch_state_dict(substate(state_dict, "ln_q"))
            self.ln_k.load_torch_state_dict(substate(state_dict, "ln_k"))
