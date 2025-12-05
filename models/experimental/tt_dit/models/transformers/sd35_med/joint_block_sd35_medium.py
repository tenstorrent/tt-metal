# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Joint Transformer Block Implementation
Direct implementation of the three joint block configurations.
"""

import ttnn
from models.experimental.tt_dit.layers.normalization import LayerNorm
from models.experimental.tt_dit.layers.feedforward import FeedForward
from models.experimental.tt_dit.layers.adalayernorm import (
    AdaLayerNormZero,
    AdaLayerNormContinuous,
    SD35AdaLayerNormZeroX,
)
from models.experimental.tt_dit.utils.substate import substate
from .attention_sd35_medium import SD35MediumSelfAttention


class SD35MediumJointTransformerBlock:
    """Joint Transformer Block for SD3.5 Medium with three configurations."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        pre_only: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: str = "rms",
        mesh_device=None,
        block_idx: int = 0,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mesh_device = mesh_device
        self.block_idx = block_idx
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Determine block configuration
        is_dual_attention = block_idx <= 12  # Blocks 0-12
        is_last_block = block_idx == 23  # Block 23

        # Initialize norm1 based on block type
        if is_dual_attention:
            # Blocks 0-12: SD35AdaLayerNormZeroX (13824 output for 1536 input)
            self.norm1 = SD35AdaLayerNormZeroX(
                hidden_size,
                hidden_size * 9,  # 13824 for 1536 input
                bias=True,
                mesh_device=mesh_device,
            )
        else:
            # Blocks 13-23: AdaLayerNormZero (9216 output for 1536 input)
            self.norm1 = AdaLayerNormZero(
                hidden_size,
                hidden_size * 6,  # 9216 for 1536 input
                bias=True,
                mesh_device=mesh_device,
            )

        # Initialize norm1_context based on block type
        if is_last_block:
            # Block 23: AdaLayerNormContinuous (3072 output for 1536 input)
            self.norm1_context = AdaLayerNormContinuous(
                hidden_size,
                hidden_size * 2,  # 3072 for 1536 input
                bias=True,
                mesh_device=mesh_device,
            )
        else:
            # Blocks 0-22: AdaLayerNormZero (9216 output for 1536 input)
            self.norm1_context = AdaLayerNormZero(
                hidden_size,
                hidden_size * 6,  # 9216 for 1536 input
                bias=True,
                mesh_device=mesh_device,
            )

        # Main attention (attn)
        self.attn = SD35MediumSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=False,
            qk_norm=qk_norm,
            mesh_device=mesh_device,
        )

        # Second attention for dual attention blocks (attn2)
        if is_dual_attention:
            self.attn2 = SD35MediumSelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                pre_only=False,
                qk_norm=qk_norm,
                mesh_device=mesh_device,
            )

        # Normalization layers
        self.norm2 = LayerNorm(
            hidden_size,
            eps=1e-6,
            elementwise_affine=False,
            mesh_device=mesh_device,
        )

        if not is_last_block:
            self.norm2_context = LayerNorm(
                hidden_size,
                eps=1e-6,
                elementwise_affine=False,
                mesh_device=mesh_device,
            )

        # Feedforward layers
        ff_hidden_dim = int(hidden_size * mlp_ratio)
        self.ff = FeedForward(
            hidden_size,
            ff_hidden_dim,
            multiple_of=256,
            ffn_dim_multiplier=None,
            activation_fn="gelu",
            bias=True,
            mesh_device=mesh_device,
        )

        if not is_last_block:
            self.ff_context = FeedForward(
                hidden_size,
                ff_hidden_dim,
                multiple_of=256,
                ffn_dim_multiplier=None,
                activation_fn="gelu",
                bias=True,
                mesh_device=mesh_device,
            )

        # Compute config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, context, x, c):
        """
        Forward pass.
        Args:
            context: [1, B, context_seq_len, hidden_size]
            x: [1, B, x_seq_len, hidden_size]
            c: [1, B, hidden_size] - conditioning
        Returns:
            context_output: [1, B, context_seq_len, hidden_size] or None
            x_output: [1, B, x_seq_len, hidden_size]
        """
        B = context.shape[1]
        context_seq_len = context.shape[2]
        x_seq_len = x.shape[2]

        # Apply norm1 to x
        x_normed, x_scale = self.norm1(x, c)

        # Apply norm1_context to context
        if self.block_idx == 23:
            # Last block uses AdaLayerNormContinuous
            context_normed, context_scale = self.norm1_context(context, c)
        else:
            # Other blocks use AdaLayerNormZero
            context_normed, context_scale = self.norm1_context(context, c)

        # Joint attention using context as added input
        context_attn, x_attn = self.attn(x_normed, x_seq_len, added_input=context_normed, added_seq_len=context_seq_len)

        # Apply norm2 to x attention output
        x_normed2 = self.norm2(x_attn)
        x_output = x + x_normed2 * x_scale

        # Apply norm2_context to context attention output
        if self.block_idx != 23:
            context_normed2 = self.norm2_context(context_attn)
            context_output = context + context_normed2 * context_scale
        else:
            context_output = None

        # Feedforward for x
        x_normed3 = self.norm2(x_output)
        x_ff = self.ff(x_normed3)
        x_output = x_output + x_ff

        # Feedforward for context (not in last block)
        if self.block_idx != 23:
            context_normed3 = self.norm2_context(context_output)
            context_ff = self.ff_context(context_normed3)
            context_output = context_output + context_ff

        # Second attention for dual attention blocks
        if self.block_idx <= 12:
            x_normed4, x_scale2 = self.norm1(x_output, c)
            x_attn2 = self.attn2(x_normed4, x_seq_len)

            x_normed5 = self.norm2(x_attn2)
            x_output = x_output + x_normed5 * x_scale2

        return context_output, x_output

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load normalization layers
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))
        self.norm1_context.load_torch_state_dict(substate(state_dict, "norm1_context"))
        self.norm2.load_torch_state_dict(substate(state_dict, "norm2"))

        if self.block_idx != 23:
            self.norm2_context.load_torch_state_dict(substate(state_dict, "norm2_context"))

        # Load attention layers
        self.attn.load_torch_state_dict(substate(state_dict, "attn"))

        if self.block_idx <= 12:
            self.attn2.load_torch_state_dict(substate(state_dict, "attn2"))

        # Load feedforward layers
        self.ff.load_torch_state_dict(substate(state_dict, "ff"))

        if self.block_idx != 23:
            self.ff_context.load_torch_state_dict(substate(state_dict, "ff_context"))
