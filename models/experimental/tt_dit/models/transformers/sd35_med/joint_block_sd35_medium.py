# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Separate JointTransformerBlock implementations for testing
AdaLayerNorm implementations inlined for easier debugging
"""

import ttnn
from models.experimental.tt_dit.layers.module import Module
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.normalization import LayerNorm
from models.experimental.tt_dit.models.transformers.sd35_med.attention_sd35_medium import SD35MediumSelfAttention
from models.experimental.tt_dit.utils.substate import substate


# =============================================================================
# Inlined AdaLayerNorm Implementations
# =============================================================================


class AdaLayerNormZero(Module):
    """
    AdaLayerNormZero with 6x scaling for SD3.5 Medium.

    Diffusers implementation:
    ```python
    class AdaLayerNormZero(nn.Module):
        def __init__(self, embedding_dim, num_embeddings=None, norm_type="layer_norm", bias=True):
            self.silu = nn.SiLU()
            self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

        def forward(self, x, emb=None):
            emb = self.linear(self.silu(emb))  # Note: SiLU BEFORE linear!
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    ```
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning (hidden_size -> 6*hidden_size)
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # Layer norm (no learnable params)
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size] (temb after passing through time embedding)
        Returns:
            normalized_x: Normalized input [1, B, seq_len, hidden_size]
            scale: Modulation params [1, B, 6, hidden_size]
        """
        # Apply layer norm to input
        normalized_x = self.norm(x)

        # Process conditioning: SiLU FIRST, then linear (matching Diffusers)
        c_silu = ttnn.silu(c)
        c_processed = self.linear(c_silu)

        # Reshape to [1, B, 6, hidden_size]
        scale = ttnn.reshape(c_processed, (1, c_processed.shape[1], 6, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


class AdaLayerNormContinuous(Module):
    """
    AdaLayerNormContinuous with 2x scaling for SD3.5 Medium block 23.

    Diffusers implementation applies shift and scale directly:
    x = self.norm(x) * (1 + scale) + shift
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # Layer norm
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size]
        Returns:
            normalized_x: Normalized input
            scale: Modulation params [1, B, 2, hidden_size] (shift, scale)
        """
        # Apply layer norm
        normalized_x = self.norm(x)

        # Process conditioning: SiLU FIRST, then linear
        c_silu = ttnn.silu(c)
        c_processed = self.linear(c_silu)

        # Reshape to [1, B, 2, hidden_size]
        scale = ttnn.reshape(c_processed, (1, c_processed.shape[1], 2, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


class SD35AdaLayerNormZeroX(Module):
    """
    SD35AdaLayerNormZeroX with 9x scaling for SD3.5 Medium blocks 0-12.

    This is used for hidden_states (x) in early blocks.
    Outputs 9 modulation params: 3 for attn, 3 for mlp, 3 for attn2.

    Diffusers implementation:
    ```python
    class SD35AdaLayerNormZeroX(nn.Module):
        def __init__(self, embedding_dim, norm_type="layer_norm", bias=True):
            self.silu = nn.SiLU()
            self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

        def forward(self, x, emb):
            emb = self.linear(self.silu(emb))  # SiLU BEFORE linear!
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(9, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2
    ```
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning (hidden_size -> 9*hidden_size)
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # Layer norm (no learnable params)
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size]
        Returns:
            normalized_x: Normalized input [1, B, seq_len, hidden_size]
            scale: Modulation params [1, B, 9, hidden_size]
                   Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2
        """
        # Apply layer norm to input
        normalized_x = self.norm(x)

        # Process conditioning: SiLU FIRST, then linear (matching Diffusers!)
        c_silu = ttnn.silu(c)
        c_processed = self.linear(c_silu)

        # Reshape to [1, B, 9, hidden_size]
        scale = ttnn.reshape(c_processed, (1, c_processed.shape[1], 9, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


# =============================================================================
# Joint Transformer Block Implementations
# =============================================================================


class FeedForward(Module):
    """FeedForward network with GELU activation."""

    def __init__(self, dim, hidden_dim, mesh_device=None):
        super().__init__()
        self.mesh_device = mesh_device

        # Linear layers
        self.net = [
            Linear(dim, hidden_dim, bias=True, mesh_device=mesh_device),
            None,  # Dropout placeholder
            Linear(hidden_dim, dim, bias=True, mesh_device=mesh_device),
        ]

    def forward(self, x):
        # First linear with GELU
        x = self.net[0](x)
        x = ttnn.gelu(x)

        # Second linear
        x = self.net[2](x)
        return x

    def load_torch_state_dict(self, state_dict):
        self.net[0].load_torch_state_dict(substate(state_dict, "net.0.proj"))
        self.net[2].load_torch_state_dict(substate(state_dict, "net.2"))


class JointTransformerBlockEarly(Module):
    """JointTransformerBlock for blocks 0-12 (SD35AdaLayerNormZeroX + AdaLayerNormZero)"""

    def __init__(self, dim: int = 1536, num_heads: int = 24, mesh_device=None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mesh_device = mesh_device

        # Blocks 0-12: SD35AdaLayerNormZeroX + AdaLayerNormZero
        self.norm1 = SD35AdaLayerNormZeroX(
            hidden_size=dim, conditioning_size=13824, bias=True, mesh_device=mesh_device, eps=eps  # 9x scaling
        )
        self.norm1_context = AdaLayerNormZero(
            hidden_size=dim, conditioning_size=9216, bias=True, mesh_device=mesh_device, eps=eps  # 6x scaling
        )

        # Attention layers
        self.attn = SD35MediumSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            pre_only=False,
            qk_norm="rms",
            eps=eps,
            mesh_device=mesh_device,
            added_proj_dim=dim,
        )

        self.attn2 = SD35MediumSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            pre_only=False,
            qk_norm="rms",
            eps=eps,
            mesh_device=mesh_device,
            added_proj_dim=None,
        )

        # Post-attention normalization
        self.norm2 = LayerNorm(dim, norm_eps=eps, norm_elementwise_affine=False, mesh_device=mesh_device)
        self.norm2_context = LayerNorm(dim, norm_eps=eps, norm_elementwise_affine=False, mesh_device=mesh_device)

        # Feed forward networks
        self.ff = FeedForward(dim=dim, hidden_dim=6144, mesh_device=mesh_device)
        self.ff_context = FeedForward(dim=dim, hidden_dim=6144, mesh_device=mesh_device)

    def forward(self, x, context, conditioning, seq_len, context_seq_len):
        """Forward pass for early blocks (0-12)

        SD35AdaLayerNormZeroX outputs 9 modulation params:
        - 0-2: shift_msa, scale_msa, gate_msa (for joint attn)
        - 3-5: shift_mlp, scale_mlp, gate_mlp (for feedforward)
        - 6-8: shift_msa2, scale_msa2, gate_msa2 (for self-attn2)

        AdaLayerNormZero outputs 6 modulation params:
        - 0-2: shift_msa, scale_msa, gate_msa (for joint attn)
        - 3-5: shift_mlp, scale_mlp, gate_mlp (for feedforward)
        """
        # First normalization with conditioning
        x_norm, x_scale = self.norm1(x, conditioning)
        context_norm, context_scale = self.norm1_context(context, conditioning)

        # Extract x modulation params (9 total for SD35AdaLayerNormZeroX)
        x_shift_msa = x_scale[:, :, 0:1, :]
        x_scale_msa = x_scale[:, :, 1:2, :]
        x_gate_msa = x_scale[:, :, 2:3, :]
        x_shift_mlp = x_scale[:, :, 3:4, :]
        x_scale_mlp = x_scale[:, :, 4:5, :]
        x_gate_mlp = x_scale[:, :, 5:6, :]
        x_shift_msa2 = x_scale[:, :, 6:7, :]
        x_scale_msa2 = x_scale[:, :, 7:8, :]
        x_gate_msa2 = x_scale[:, :, 8:9, :]

        # Extract context modulation params (6 total for AdaLayerNormZero)
        c_shift_msa = context_scale[:, :, 0:1, :]
        c_scale_msa = context_scale[:, :, 1:2, :]
        c_gate_msa = context_scale[:, :, 2:3, :]
        c_shift_mlp = context_scale[:, :, 3:4, :]
        c_scale_mlp = context_scale[:, :, 4:5, :]
        c_gate_mlp = context_scale[:, :, 5:6, :]

        # Apply modulation for joint attention
        x_modulated = x_norm * (1 + x_scale_msa) + x_shift_msa
        context_modulated = context_norm * (1 + c_scale_msa) + c_shift_msa

        # Joint attention (attn) - x and context attend together
        attn_out, context_attn_out = self.attn(
            x_modulated, seq_len, added_input=context_modulated, added_seq_len=context_seq_len
        )

        # Apply gate and residual for joint attention (NO silu on gate!)
        x = x + x_gate_msa * attn_out
        context = context + c_gate_msa * context_attn_out

        # Second attention (attn2) - self-attention on x only
        x_norm2 = self.norm2(x)
        x_modulated2 = x_norm2 * (1 + x_scale_msa2) + x_shift_msa2
        attn2_out = self.attn2(x_modulated2, seq_len)
        x = x + x_gate_msa2 * attn2_out

        # Feed forward
        x_norm_ff = self.norm2(x)
        context_norm_ff = self.norm2_context(context)

        x_modulated_ff = x_norm_ff * (1 + x_scale_mlp) + x_shift_mlp
        context_modulated_ff = context_norm_ff * (1 + c_scale_mlp) + c_shift_mlp

        ff_out = self.ff(x_modulated_ff)
        context_ff_out = self.ff_context(context_modulated_ff)

        x = x + x_gate_mlp * ff_out
        context = context + c_gate_mlp * context_ff_out

        return x, context

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))
        self.norm1_context.load_torch_state_dict(substate(state_dict, "norm1_context"))
        # norm2 and norm2_context have norm_elementwise_affine=False, no weights to load
        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.attn2.load_state_dict(substate(state_dict, "attn2"))
        self.ff.load_torch_state_dict(substate(state_dict, "ff"))
        self.ff_context.load_torch_state_dict(substate(state_dict, "ff_context"))


class JointTransformerBlockMiddle(Module):
    """JointTransformerBlock for blocks 13-22 (AdaLayerNormZero + AdaLayerNormZero)

    Middle blocks do NOT have attn2 - only joint attention (attn).
    """

    def __init__(self, dim: int = 1536, num_heads: int = 24, mesh_device=None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mesh_device = mesh_device

        # Blocks 13-22: AdaLayerNormZero + AdaLayerNormZero
        self.norm1 = AdaLayerNormZero(
            hidden_size=dim, conditioning_size=9216, bias=True, mesh_device=mesh_device, eps=eps  # 6x scaling
        )
        self.norm1_context = AdaLayerNormZero(
            hidden_size=dim, conditioning_size=9216, bias=True, mesh_device=mesh_device, eps=eps  # 6x scaling
        )

        # Only joint attention (no attn2 in middle blocks!)
        self.attn = SD35MediumSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            pre_only=False,
            qk_norm="rms",
            eps=eps,
            mesh_device=mesh_device,
            added_proj_dim=dim,
        )

        # Post-attention normalization
        self.norm2 = LayerNorm(dim, norm_eps=eps, norm_elementwise_affine=False, mesh_device=mesh_device)
        self.norm2_context = LayerNorm(dim, norm_eps=eps, norm_elementwise_affine=False, mesh_device=mesh_device)

        # Feed forward networks
        self.ff = FeedForward(dim=dim, hidden_dim=6144, mesh_device=mesh_device)
        self.ff_context = FeedForward(dim=dim, hidden_dim=6144, mesh_device=mesh_device)

    def forward(self, x, context, conditioning, seq_len, context_seq_len):
        """Forward pass for middle blocks (13-22)

        Middle blocks use AdaLayerNormZero (6 params) for both x and context.
        Unlike early blocks, middle blocks do NOT use attn2 in forward pass.

        AdaLayerNormZero outputs 6 modulation params:
        - 0-2: shift_msa, scale_msa, gate_msa (for joint attn)
        - 3-5: shift_mlp, scale_mlp, gate_mlp (for feedforward)
        """
        # First normalization with conditioning
        x_norm, x_scale = self.norm1(x, conditioning)
        context_norm, context_scale = self.norm1_context(context, conditioning)

        # Extract x modulation params (6 total for AdaLayerNormZero)
        x_shift_msa = x_scale[:, :, 0:1, :]
        x_scale_msa = x_scale[:, :, 1:2, :]
        x_gate_msa = x_scale[:, :, 2:3, :]
        x_shift_mlp = x_scale[:, :, 3:4, :]
        x_scale_mlp = x_scale[:, :, 4:5, :]
        x_gate_mlp = x_scale[:, :, 5:6, :]

        # Extract context modulation params (6 total for AdaLayerNormZero)
        c_shift_msa = context_scale[:, :, 0:1, :]
        c_scale_msa = context_scale[:, :, 1:2, :]
        c_gate_msa = context_scale[:, :, 2:3, :]
        c_shift_mlp = context_scale[:, :, 3:4, :]
        c_scale_mlp = context_scale[:, :, 4:5, :]
        c_gate_mlp = context_scale[:, :, 5:6, :]

        # Apply modulation for joint attention
        x_modulated = x_norm * (1 + x_scale_msa) + x_shift_msa
        context_modulated = context_norm * (1 + c_scale_msa) + c_shift_msa

        # Joint attention (attn) - x and context attend together
        attn_out, context_attn_out = self.attn(
            x_modulated, seq_len, added_input=context_modulated, added_seq_len=context_seq_len
        )

        # Apply gate and residual for joint attention (NO silu on gate!)
        x = x + x_gate_msa * attn_out
        context = context + c_gate_msa * context_attn_out

        # NOTE: Middle blocks do NOT use attn2 - only early blocks (0-12) use it

        # Feed forward
        x_norm_ff = self.norm2(x)
        context_norm_ff = self.norm2_context(context)

        x_modulated_ff = x_norm_ff * (1 + x_scale_mlp) + x_shift_mlp
        context_modulated_ff = context_norm_ff * (1 + c_scale_mlp) + c_shift_mlp

        ff_out = self.ff(x_modulated_ff)
        context_ff_out = self.ff_context(context_modulated_ff)

        x = x + x_gate_mlp * ff_out
        context = context + c_gate_mlp * context_ff_out

        return x, context

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))
        self.norm1_context.load_torch_state_dict(substate(state_dict, "norm1_context"))
        # norm2 and norm2_context have norm_elementwise_affine=False, no weights to load
        self.attn.load_state_dict(substate(state_dict, "attn"))
        # NOTE: Middle blocks do NOT have attn2
        self.ff.load_torch_state_dict(substate(state_dict, "ff"))
        self.ff_context.load_torch_state_dict(substate(state_dict, "ff_context"))


class JointTransformerBlockFinal(Module):
    """JointTransformerBlock for block 23 (AdaLayerNormZero + AdaLayerNormContinuous)"""

    def __init__(self, dim: int = 1536, num_heads: int = 24, mesh_device=None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mesh_device = mesh_device

        # Block 23: AdaLayerNormZero + AdaLayerNormContinuous
        self.norm1 = AdaLayerNormZero(
            hidden_size=dim, conditioning_size=9216, bias=True, mesh_device=mesh_device, eps=eps  # 6x scaling
        )
        self.norm1_context = AdaLayerNormContinuous(
            hidden_size=dim, conditioning_size=3072, bias=True, mesh_device=mesh_device, eps=eps  # 2x scaling
        )

        # Attention layers
        self.attn = SD35MediumSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            pre_only=False,
            qk_norm="rms",
            eps=eps,
            mesh_device=mesh_device,
            added_proj_dim=dim,
        )

        self.attn2 = SD35MediumSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            pre_only=False,
            qk_norm="rms",
            eps=eps,
            mesh_device=mesh_device,
            added_proj_dim=None,
        )

        # Post-attention normalization
        self.norm2 = LayerNorm(dim, norm_eps=eps, norm_elementwise_affine=False, mesh_device=mesh_device)

        # Feed forward network (only main, no context)
        self.ff = FeedForward(dim=dim, hidden_dim=6144, mesh_device=mesh_device)

    def forward(self, x, context, conditioning, seq_len, context_seq_len):
        """Forward pass for final block"""
        # First normalization with conditioning
        x_norm, scale = self.norm1(x, conditioning)
        context_norm, context_scale = self.norm1_context(context, conditioning)

        # Extract scale shifts (6x for final block)
        shift_msa = scale[:, :, 0]
        scale_msa = scale[:, :, 1]
        gate_msa = scale[:, :, 2]
        shift_mlp = scale[:, :, 3]
        scale_mlp = scale[:, :, 4]
        gate_mlp = scale[:, :, 5]

        # Context scales (2x)
        context_shift_msa = context_scale[:, :, 0]
        context_scale_msa = context_scale[:, :, 1]

        # Apply modulation
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        context_modulated = context_norm * (1 + context_scale_msa) + context_shift_msa

        # First attention with context
        attn_out, context_attn_out = self.attn(
            x_modulated, seq_len, added_input=context_modulated, added_seq_len=context_seq_len
        )

        # Apply gate modulation
        attn_out = attn_out * ttnn.silu(gate_msa)
        context_attn_out = context_attn_out * ttnn.silu(context_scale_msa)  # Use scale as gate for context

        # Residual connections
        x = x + attn_out
        context = context + context_attn_out

        # Second normalization
        x_norm = self.norm2(x)

        # Apply MLP modulation
        x_modulated = x_norm * (1 + scale_mlp) + shift_mlp

        # Feed forward
        ff_out = self.ff(x_modulated)
        ff_out = ff_out * ttnn.silu(gate_mlp)

        # Residual connection
        x = x + ff_out

        return x, context

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))
        self.norm1_context.load_torch_state_dict(substate(state_dict, "norm1_context"))
        # norm2 has norm_elementwise_affine=False, no weights to load
        self.attn.load_state_dict(substate(state_dict, "attn"))
        self.attn2.load_state_dict(substate(state_dict, "attn2"))
        self.ff.load_torch_state_dict(substate(state_dict, "ff"))
