# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3Transformer2DModel implementation for SD3.5 Medium
"""

import ttnn
from models.experimental.tt_dit.layers.module import Module
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.normalization import LayerNorm
from models.experimental.tt_dit.utils.substate import substate

# Import working implementations from embeddings
from models.experimental.tt_dit.layers.embeddings import (
    PatchEmbed,
    SD35CombinedTimestepTextProjEmbeddings,
)

# Import JointTransformerBlock implementations
from models.experimental.tt_dit.models.transformers.sd35_med.joint_block_sd35_medium import (
    JointTransformerBlockEarly,
    JointTransformerBlockMiddle,
    JointTransformerBlockFinal,
)


class AdaLayerNormContinuousOutput(Module):
    """
    AdaLayerNormContinuous for norm_out - applies modulation directly.

    Diffusers implementation:
    ```python
    emb = self.linear(self.silu(conditioning))
    scale, shift = emb.chunk(2, dim=-1)  # NOTE: scale first, then shift!
    x = self.norm(x) * (1 + scale) + shift
    return x  # Returns modulated tensor directly
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

        # Linear layer: hidden_size -> conditioning_size (2 * hidden_size)
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
            c: Conditioning tensor [1, B, 1, hidden_size] (temb)
        Returns:
            Modulated tensor [1, B, seq_len, hidden_size]
        """
        # Apply layer norm
        x_norm = self.norm(x)

        # Process conditioning: SiLU FIRST, then linear (matching Diffusers)
        c_silu = ttnn.silu(c)
        emb = self.linear(c_silu)

        # Reshape to [1, B, 2, hidden_size] for chunking
        emb = ttnn.reshape(emb, (1, emb.shape[1], 2, self.hidden_size))

        # Extract scale and shift - Diffusers: scale, shift = emb.chunk(2)
        # So index 0 = scale, index 1 = shift
        scale = emb[:, :, 0:1, :]
        shift = emb[:, :, 1:2, :]

        # Apply modulation: x = norm(x) * (1 + scale) + shift
        x_out = x_norm * (1 + scale) + shift

        return x_out

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


# ============ Main Transformer Model ============


class SD3Transformer2DModel(Module):
    """SD3Transformer2DModel for SD3.5 Medium with 24 transformer blocks"""

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 2432,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 192,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.mesh_device = mesh_device

        # Calculate hidden dimension
        self.inner_dim = num_attention_heads * attention_head_dim

        # Positional embedding (patch embedding) - use imported version
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            tp_mesh_axis=None,
            sp_mesh_axis=None,
            mesh_device=mesh_device,
        )

        # Time and text embedding - use imported working version
        self.time_text_embed = SD35CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            mesh_device=mesh_device,
        )

        # Context embedder
        self.context_embedder = Linear(
            in_features=joint_attention_dim,
            out_features=self.inner_dim,
            bias=True,
            mesh_device=mesh_device,
        )

        # Transformer blocks
        self.transformer_blocks = []

        # Blocks 0-12: Early blocks with SD35AdaLayerNormZeroX + AdaLayerNormZero
        for i in range(13):
            block = JointTransformerBlockEarly(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                mesh_device=mesh_device,
                eps=eps,
            )
            self.transformer_blocks.append(block)

        # Blocks 13-22: Middle blocks with AdaLayerNormZero + AdaLayerNormZero
        for i in range(10):
            block = JointTransformerBlockMiddle(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                mesh_device=mesh_device,
                eps=eps,
            )
            self.transformer_blocks.append(block)

        # Block 23: Final block with AdaLayerNormZero + AdaLayerNormContinuous
        block = JointTransformerBlockFinal(
            dim=self.inner_dim,
            num_heads=num_attention_heads,
            mesh_device=mesh_device,
            eps=eps,
        )
        self.transformer_blocks.append(block)

        # Output normalization - use inlined version with correct scale/shift order
        self.norm_out = AdaLayerNormContinuousOutput(
            hidden_size=self.inner_dim,
            conditioning_size=self.inner_dim * 2,  # 2x for scale and shift
            bias=True,
            mesh_device=mesh_device,
            eps=eps,
        )

        self.proj_out = Linear(
            in_features=self.inner_dim,
            out_features=out_channels * patch_size * patch_size,
            bias=True,
            mesh_device=mesh_device,
        )

        # Compute kernel config for attention
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.core_grid = mesh_device.compute_with_storage_grid_size()

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        timestep: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        seq_len: int,
        context_seq_len: int,
    ):
        """
        Forward pass of SD3Transformer2DModel

        Args:
            hidden_states: Input latent tensor [B, H, W, C] NHWC format
            encoder_hidden_states: Text encoder output [B, context_seq_len, joint_attention_dim]
            timestep: Timestep tensor [B, 1]
            pooled_projection: Pooled text projection [1, 1, B, pooled_projection_dim]
            seq_len: Sequence length for hidden states
            context_seq_len: Sequence length for context

        Returns:
            Output latent tensor [1, B, seq_len, out_channels*patch_size^2]
        """
        # Apply positional embedding (also handles patchify)
        hidden_states = self.pos_embed(hidden_states)

        # Add batch dimension for TTNN convention [1, B, seq_len, dim]
        if hidden_states.shape[0] != 1 or len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(
                hidden_states, (1, hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2])
            )

        # Compute time and text embeddings
        temb = self.time_text_embed(timestep, pooled_projection)

        # Embed context
        context = self.context_embedder(encoder_hidden_states)

        # Add batch dimension for context if needed
        if len(context.shape) == 3:
            context = ttnn.reshape(context, (1, context.shape[0], context.shape[1], context.shape[2]))

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            hidden_states, context = block(
                x=hidden_states,
                context=context,
                conditioning=temb,
                seq_len=seq_len,
                context_seq_len=context_seq_len,
            )

        # Apply output normalization (applies modulation internally)
        hidden_states = self.norm_out(hidden_states, temb)

        # Final projection
        output = self.proj_out(hidden_states)

        return output

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict"""
        # Load positional embedding
        self.pos_embed.load_torch_state_dict(substate(state_dict, "pos_embed"))

        # Load time and text embeddings
        self.time_text_embed.load_torch_state_dict(substate(state_dict, "time_text_embed"))

        # Load context embedder
        self.context_embedder.load_torch_state_dict(substate(state_dict, "context_embedder"))

        # Load transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block.load_torch_state_dict(substate(state_dict, f"transformer_blocks.{i}"))

        # Load output layers
        self.norm_out.load_torch_state_dict(substate(state_dict, "norm_out"))
        self.proj_out.load_torch_state_dict(substate(state_dict, "proj_out"))

    def to_cached_state_dict(self, path_prefix):
        """Create cached state dict for faster loading"""
        cache_dict = {}

        # Cache positional embedding
        pos_embed_cache = self.pos_embed.to_cached_state_dict(path_prefix + "pos_embed.")
        for key, value in pos_embed_cache.items():
            cache_dict[f"pos_embed.{key}"] = value

        # Cache time and text embeddings
        time_text_cache = self.time_text_embed.to_cached_state_dict(path_prefix + "time_text_embed.")
        for key, value in time_text_cache.items():
            cache_dict[f"time_text_embed.{key}"] = value

        # Cache context embedder
        context_cache = self.context_embedder.to_cached_state_dict(path_prefix + "context_embedder.")
        for key, value in context_cache.items():
            cache_dict[f"context_embedder.{key}"] = value

        # Cache transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block_cache = block.to_cached_state_dict(path_prefix + f"transformer_blocks.{i}.")
            for key, value in block_cache.items():
                cache_dict[f"transformer_blocks.{i}.{key}"] = value

        # Cache output layers
        norm_out_cache = self.norm_out.to_cached_state_dict(path_prefix + "norm_out.")
        proj_out_cache = self.proj_out.to_cached_state_dict(path_prefix + "proj_out.")

        for key, value in norm_out_cache.items():
            cache_dict[f"norm_out.{key}"] = value
        for key, value in proj_out_cache.items():
            cache_dict[f"proj_out.{key}"] = value

        return cache_dict
