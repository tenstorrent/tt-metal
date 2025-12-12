# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3Transformer2DModel implementation for SD3.5 Medium
"""

import math
import torch
import ttnn
from models.experimental.tt_dit.layers.module import Module, Parameter
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.normalization import LayerNorm
from models.experimental.tt_dit.utils.substate import substate

# Import working implementations from embeddings
from models.experimental.tt_dit.layers.embeddings import (
    SD35CombinedTimestepTextProjEmbeddings,
)


class SD35PatchEmbed(Module):
    """
    Custom PatchEmbed for SD3.5 Medium that handles 4D output tensors properly.
    Converts image latents to patch embeddings with positional encoding.
    Supports dynamic input sizes by cropping pos_embed at runtime.
    """

    def __init__(
        self,
        height,
        width,
        patch_size,
        in_channels,
        embed_dim,
        pos_embed_max_size,
        mesh_device=None,
    ):
        super().__init__()

        self.height = height // patch_size  # Max height in patches
        self.width = width // patch_size  # Max width in patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.pos_embed_max_size = pos_embed_max_size
        self.mesh_device = mesh_device

        # Compute kernel config for linear operations
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Projection weights (equivalent to conv2d)
        self.proj_weight = Parameter(
            total_shape=[in_channels * patch_size * patch_size, embed_dim],
            mesh_axes=[None, None],
            device=mesh_device,
        )
        self.proj_bias = Parameter(
            total_shape=[1, 1, 1, embed_dim],
            mesh_axes=[None, None, None, None],
            device=mesh_device,
        )

        # Position embeddings - store at max size [1, 1, max_seq_len, embed_dim]
        # Will be cropped at runtime based on actual input size
        max_seq_len = self.height * self.width
        self.pos_embed = Parameter(
            total_shape=[1, 1, max_seq_len, embed_dim],
            device=mesh_device,
            mesh_axes=[None, None, None, None],
        )

        # Store the torch pos_embed for runtime cropping
        self._torch_pos_embed = None

    def _cropped_pos_embed_torch(self, pos_embed_param, target_height, target_width):
        """Crop position embeddings from max size to target size (torch tensor)."""
        pos_embed_max_size = math.isqrt(pos_embed_param.shape[1])
        top = (pos_embed_max_size - target_height) // 2
        left = (pos_embed_max_size - target_width) // 2

        spatial_pos_embed = pos_embed_param.reshape([1, pos_embed_max_size, pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + target_height, left : left + target_width, :]
        spatial_pos_embed = spatial_pos_embed.reshape([1, 1, -1, spatial_pos_embed.shape[-1]])

        return spatial_pos_embed

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare PyTorch state dict for loading."""
        conv_weight = state.pop("proj.weight", None)
        if conv_weight is not None:
            # Convert from (out_channels, in_channels, kh, kw) to (kh*kw*in_channels, out_channels)
            out_channels, in_c, kh, kw = conv_weight.shape
            conv_weight = conv_weight.permute(2, 3, 1, 0)  # (kh, kw, in_c, out_channels)
            conv_weight = conv_weight.reshape(kh * kw * in_c, out_channels)
            state["proj_weight"] = conv_weight

        if "proj.bias" in state:
            state["proj_bias"] = state.pop("proj.bias").reshape(1, 1, 1, -1)

        if "pos_embed" in state:
            # Store original pos_embed for runtime cropping
            self._torch_pos_embed = state["pos_embed"].clone()
            # Crop for default size
            state["pos_embed"] = self._cropped_pos_embed_torch(state.pop("pos_embed"), self.height, self.width)

    def forward(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: apply patch projection and add position embeddings.

        Args:
            latent: Input tensor of shape (batch_size, height, width, channels) in NHWC format

        Returns:
            Patch embeddings of shape (1, batch_size, num_patches, embed_dim)
        """
        batch_size, img_h, img_w, img_c = latent.shape
        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        latent = self._unfold_conv2d(latent)

        # Check if we need to use different pos_embed size
        actual_seq_len = patches_h * patches_w
        stored_seq_len = self.pos_embed.data.shape[2]

        if actual_seq_len != stored_seq_len and self._torch_pos_embed is not None:
            # Crop pos_embed to match actual input size
            cropped_pos_embed = self._cropped_pos_embed_torch(self._torch_pos_embed, patches_h, patches_w)
            pos_embed_tensor = ttnn.from_torch(
                cropped_pos_embed,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return latent + pos_embed_tensor
        else:
            return latent + self.pos_embed.data

    def _unfold_conv2d(self, x):
        """
        Unfold conv2d operation: reshape patches and apply linear transformation.

        Args:
            x: Input tensor (batch_size, height, width, channels)

        Returns:
            Projected tensor (1, batch_size, patches_h * patches_w, embed_dim)
        """
        batch_size, img_h, img_w, img_c = x.shape

        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        # Reshape input to extract patches
        # (batch_size, patches_h, patch_size, patches_w, patch_size, img_c)
        x = ttnn.reshape(x, (batch_size, patches_h, self.patch_size, patches_w, self.patch_size, img_c))

        # Permute to group patch dimensions together
        # (batch_size, patches_h, patches_w, patch_size, patch_size, img_c)
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))

        # Flatten patch dimensions
        # (batch_size, patches_h * patches_w, patch_size * patch_size * img_c)
        x = ttnn.reshape(x, (batch_size, patches_h * patches_w, self.patch_size * self.patch_size * img_c))

        # Apply linear projection (equivalent to conv2d)
        out = ttnn.linear(
            x,
            self.proj_weight.data,
            bias=self.proj_bias.data,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        out = ttnn.reshape(out, (1, batch_size, patches_h * patches_w, -1))
        return out


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


# =============================================================================
# Main Transformer Model
# =============================================================================


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

        # Positional embedding (patch embedding) - use custom SD35PatchEmbed
        self.pos_embed = SD35PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
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

    def unpatchify(self, x, height, width):
        """
        Unpatchify the output tensor from [B, seq_len, patch_size^2 * out_channels]
        to [B, out_channels, height, width]

        This matches Diffusers' _get_output_for_patched_inputs which uses:
        output = hidden_states.reshape(batch_size, height, width, p, p, out_channels)
        output = output.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, out_channels, height*p, width*p)

        Args:
            x: Tensor of shape [B, seq_len, patch_size^2 * out_channels] or [1, B, seq_len, ...]
            height: Output height in latent space (before VAE upscaling)
            width: Output width in latent space (before VAE upscaling)

        Returns:
            Tensor of shape [B, out_channels, height, width]
        """
        import torch

        # Handle TTNN tensor
        if hasattr(x, "shape") and not isinstance(x, torch.Tensor):
            x = ttnn.to_torch(ttnn.from_device(x))

        # Remove extra dimensions
        while x.dim() > 3:
            x = x.squeeze(0)

        batch_size = x.shape[0]
        p = self.patch_size
        c = self.out_channels
        h = height // p
        w = width // p

        # Diffusers order: features are [p * p * c] -> reshape to [p, p, c]
        # Reshape: [B, h*w, p*p*c] -> [B, h, w, p, p, c]
        x = x.reshape(batch_size, h, w, p, p, c)

        # Permute: [B, h, w, p, p, c] -> [B, c, h, p, w, p]
        x = x.permute(0, 5, 1, 3, 2, 4)

        # Reshape: [B, c, h, p, w, p] -> [B, c, h*p, w*p]
        x = x.reshape(batch_size, c, h * p, w * p)

        return x

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
