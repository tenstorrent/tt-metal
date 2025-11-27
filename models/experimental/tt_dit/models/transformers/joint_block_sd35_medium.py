# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Joint Block Implementation
Combines context and x inputs through joint attention processing.
"""

import ttnn
from .dismantled_block_sd35_medium import SD35MediumDismantledBlock


class SD35MediumJointBlock:
    """Joint block for SD3.5 Medium with context and x processing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_only: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: str = "rms",
        x_block_self_attn: bool = False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mesh_device = mesh_device

        # Create context and x blocks
        self.context_block = SD35MediumDismantledBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            scale_mod_only=scale_mod_only,
            swiglu=swiglu,
            qk_norm=qk_norm,
            x_block_self_attn=False,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        self.x_block = SD35MediumDismantledBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            pre_only=False,
            scale_mod_only=scale_mod_only,
            swiglu=swiglu,
            qk_norm=qk_norm,
            x_block_self_attn=x_block_self_attn,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # Compute config for joint attention
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, context, x, c):
        """
        Forward pass for joint block.
        Args:
            context: [1, B, context_seq_len, hidden_size]
            x: [1, B, x_seq_len, hidden_size]
            c: [1, B, hidden_size]
        Returns:
            context_output: [1, B, context_seq_len, hidden_size] or None
            x_output: [1, B, x_seq_len, hidden_size]
        """
        B = context.shape[1]
        context_seq_len = context.shape[2]
        x_seq_len = x.shape[2]

        # Get QKV projections from both blocks using pre_attention_qkv
        context_qkv, context_intermediates = self.context_block.pre_attention_qkv(context, c)

        if self.x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = self.x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = self.x_block.pre_attention_qkv(x, c)

        # Joint scaled dot-product attention using TTNN's optimized operation
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=64,
            k_chunk_size=64,
        )

        # Use TTNN's joint attention operation
        context_attn, x_attn = ttnn.transformer.joint_scaled_dot_product_attention(
            context_qkv[0],
            context_qkv[1],
            context_qkv[2],  # context Q, K, V
            x_qkv[0],
            x_qkv[1],
            x_qkv[2],  # x Q, K, V
            joint_strategy="rear",
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Reshape attention outputs from [B, num_heads, seq_len, head_dim] to [1, B, seq_len, hidden_size]
        # for post_attention processing
        context_attn = ttnn.transformer.concatenate_heads(context_attn)
        x_attn = ttnn.transformer.concatenate_heads(x_attn)

        # Apply post-attention processing
        if not self.context_block.pre_only:
            context_output = self.context_block.post_attention(context_attn, *context_intermediates)
        else:
            context_output = None

        if self.x_block.x_block_self_attn:
            # Second attention for x_block self-attention mode
            x_attn2 = ttnn.transformer.scaled_dot_product_attention(
                x_qkv2[0],
                x_qkv2[1],
                x_qkv2[2],
                is_causal=False,
                scale=self.x_block.attn2.scale,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            # Reshape second attention output as well
            x_attn2 = ttnn.transformer.concatenate_heads(x_attn2)
            x_output = self.x_block.post_attention_x(x_attn, x_attn2, *x_intermediates)
        else:
            x_output = self.x_block.post_attention(x_attn, *x_intermediates)

        return context_output, x_output

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Split state dict for context and x blocks
        context_state_dict = {}
        x_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("context_block."):
                context_state_dict[key[14:]] = value  # Remove "context_block." prefix
            elif key.startswith("x_block."):
                x_state_dict[key[8:]] = value  # Remove "x_block." prefix

        self.context_block.load_state_dict(context_state_dict)
        self.x_block.load_state_dict(x_state_dict)
