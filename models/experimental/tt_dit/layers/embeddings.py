# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn

from ..utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ..utils.substate import substate
from .linear import Linear


# Helper classes for SD35Transformer2DModel
class TimestepEmbedding:
    def __init__(self, in_channels, time_embed_dim, mesh_device=None):
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.mesh_device = mesh_device

        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True, mesh_device=mesh_device)
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True, mesh_device=mesh_device)

    def to_cached_state_dict(self, path_prefix):
        linear_1_cache = self.linear_1.to_cached_state_dict(path_prefix + "linear_1.")
        linear_2_cache = self.linear_2.to_cached_state_dict(path_prefix + "linear_2.")
        cache_dict = {}
        # Add linear_1. prefix to all keys from linear_1_cache
        for key, value in linear_1_cache.items():
            cache_dict[f"linear_1.{key}"] = value
        # Add linear_2. prefix to all keys from linear_2_cache
        for key, value in linear_2_cache.items():
            cache_dict[f"linear_2.{key}"] = value
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.linear_1.from_cached_state_dict(substate(cache_dict, "linear_1"))
        self.linear_2.from_cached_state_dict(substate(cache_dict, "linear_2"))

    def load_state_dict(self, state_dict):
        self.linear_1.load_state_dict(substate(state_dict, "linear_1"))
        self.linear_2.load_state_dict(substate(state_dict, "linear_2"))

    def __call__(self, x):
        x = self.linear_1(x)
        x = ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self.linear_2(x)


class PixartAlphaTextProjection:
    def __init__(self, in_features, hidden_size, mesh_device=None):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device

        self.linear_1 = Linear(in_features, hidden_size, bias=True, mesh_device=mesh_device)
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True, mesh_device=mesh_device)

    def to_cached_state_dict(self, path_prefix):
        linear_1_cache = self.linear_1.to_cached_state_dict(path_prefix + "linear_1.")
        linear_2_cache = self.linear_2.to_cached_state_dict(path_prefix + "linear_2.")
        cache_dict = {}
        # Add linear_1. prefix to all keys from linear_1_cache
        for key, value in linear_1_cache.items():
            cache_dict[f"linear_1.{key}"] = value
        # Add linear_2. prefix to all keys from linear_2_cache
        for key, value in linear_2_cache.items():
            cache_dict[f"linear_2.{key}"] = value
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.linear_1.from_cached_state_dict(substate(cache_dict, "linear_1"))
        self.linear_2.from_cached_state_dict(substate(cache_dict, "linear_2"))

    def load_state_dict(self, state_dict):
        self.linear_1.load_state_dict(substate(state_dict, "linear_1"))
        self.linear_2.load_state_dict(substate(state_dict, "linear_2"))

    def __call__(self, x):
        x = self.linear_1(x)
        x = ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self.linear_2(x)


class SD35CombinedTimestepTextProjEmbeddings:
    def __init__(self, embedding_dim, pooled_projection_dim, mesh_device=None):
        self.embedding_dim = embedding_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.mesh_device = mesh_device

        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, mesh_device=mesh_device)
        self.text_embedder = PixartAlphaTextProjection(pooled_projection_dim, embedding_dim, mesh_device=mesh_device)

        self.time_proj_factor = self._create_time_proj_factor(256)

    def to_cached_state_dict(self, path_prefix):
        timestep_embedder_cache = self.timestep_embedder.to_cached_state_dict(path_prefix + "timestep_embedder.")
        text_embedder_cache = self.text_embedder.to_cached_state_dict(path_prefix + "text_embedder.")

        cache_dict = {}
        # Add timestep_embedder. prefix to all keys from timestep_embedder_cache
        for key, value in timestep_embedder_cache.items():
            cache_dict[f"timestep_embedder.{key}"] = value
        # Add text_embedder. prefix to all keys from text_embedder_cache
        for key, value in text_embedder_cache.items():
            cache_dict[f"text_embedder.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.timestep_embedder.from_cached_state_dict(substate(cache_dict, "timestep_embedder"))
        self.text_embedder.from_cached_state_dict(substate(cache_dict, "text_embedder"))

    def load_state_dict(self, state_dict):
        self.timestep_embedder.load_state_dict(substate(state_dict, "timestep_embedder"))
        self.text_embedder.load_state_dict(substate(state_dict, "text_embedder"))

    def _create_time_proj_factor(self, num_channels) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent)

        return ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=self.mesh_device))

    def __call__(self, timestep, pooled_projection):
        # Time projection (sinusoidal embedding)
        emb = timestep * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        text_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + text_emb


class PatchEmbed:
    """
    Patch embedding with unfolded conv2d implementation.
    Converts input images to patch embeddings with positional encoding.
    """

    def __init__(
        self,
        height,
        width,
        patch_size,
        in_channels,
        embed_dim,
        pos_embed_max_size,
        tp_mesh_axis,
        sp_mesh_axis,
        mesh_device=None,
    ):
        self.height = height // patch_size
        self.width = width // patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.pos_embed_max_size = pos_embed_max_size
        self.mesh_device = mesh_device
        self.tp_mesh_axis = tp_mesh_axis
        self.sp_mesh_axis = sp_mesh_axis

        # Conv2d projection weights (unfolded)
        # Weight shape: (kernel_h * kernel_w * in_channels, out_channels)
        conv_in_features = patch_size * patch_size * in_channels
        self.proj_weight = None
        self.proj_bias = None

        # Position embeddings
        self.pos_embed = None

        # Compute kernel config for linear operations
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _cropped_pos_embed(self, pos_embed_param):
        pos_embed_max_size = math.isqrt(pos_embed_param.shape[1])
        top = (pos_embed_max_size - self.height) // 2
        left = (pos_embed_max_size - self.width) // 2

        spatial_pos_embed = pos_embed_param.reshape([1, pos_embed_max_size, pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + self.height, left : left + self.width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])

    def to_cached_state_dict(self, path_prefix, path_suffix=".tensorbin"):
        cache_dict = {}

        # Cache proj_weight
        proj_weight_path = path_prefix + "proj_weight" + path_suffix
        ttnn.dump_tensor(proj_weight_path, self.proj_weight)
        cache_dict["proj_weight"] = proj_weight_path

        # Cache proj_bias if it exists
        if self.proj_bias is not None:
            proj_bias_path = path_prefix + "proj_bias" + path_suffix
            ttnn.dump_tensor(proj_bias_path, self.proj_bias)
            cache_dict["proj_bias"] = proj_bias_path

        # Cache pos_embed
        pos_embed_path = path_prefix + "pos_embed" + path_suffix
        ttnn.dump_tensor(pos_embed_path, self.pos_embed)
        cache_dict["pos_embed"] = pos_embed_path

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.proj_weight = ttnn.load_tensor(cache_dict["proj_weight"], device=self.mesh_device)
        if "proj_bias" in cache_dict:
            self.proj_bias = ttnn.load_tensor(cache_dict["proj_bias"], device=self.mesh_device)
        self.pos_embed = ttnn.load_tensor(cache_dict["pos_embed"], device=self.mesh_device)

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load conv2d projection weights
        conv_weight = state_dict["proj.weight"]
        # Convert from (out_channels, in_channels, kh, kw) to (kh*kw*in_channels, out_channels)
        out_channels, in_c, kh, kw = conv_weight.shape
        conv_weight = conv_weight.permute(2, 3, 1, 0)  # (kh, kw, in_c, out_channels)
        conv_weight = conv_weight.reshape(kh * kw * in_c, out_channels)

        self.proj_weight = bf16_tensor(
            conv_weight,
            device=self.mesh_device,
            mesh_axis=self.tp_mesh_axis,
            shard_dim=1,
        )

        if "proj.bias" in state_dict:
            bias = state_dict["proj.bias"].reshape(1, 1, 1, -1)
            self.proj_bias = bf16_tensor(
                bias,
                device=self.mesh_device,
                mesh_axis=self.tp_mesh_axis,
                shard_dim=3,
            )

        # Load position embeddings
        pos_embed_param = state_dict["pos_embed"]
        cropped_pos_embed = self._cropped_pos_embed(pos_embed_param)
        self.pos_embed = bf16_tensor_2dshard(
            cropped_pos_embed,
            device=self.mesh_device,
            shard_mapping={self.sp_mesh_axis: 1, self.tp_mesh_axis: 2},
        )

    def __call__(self, latent):
        """
        Forward pass: apply patch projection and add position embeddings.

        Args:
            latent: Input tensor of shape (batch_size, height, width, channels) in NHWC format
                fractured dim 2 along sp_axis

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
                fractured dim 2 along sp_axis, dim 3 along tp_axis
        """
        batch_size, img_h, img_w, img_c = latent.shape

        # Apply unfolded conv2d projection
        latent = self._unfold_conv2d(latent)
        return latent + self.pos_embed

    def _unfold_conv2d(self, x):
        """
        Unfold conv2d operation: reshape patches and apply linear transformation.

        Args:
            x: Input tensor (batch_size, height, width, channels)
                fractured dim 2 along sp_axis

        Returns:
            Projected tensor (1, batch_size, patches_h * patches_w, embed_dim)
                fractured dim 2 along sp_axis, dim 3 along tp_axis
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
            self.proj_weight,
            bias=self.proj_bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        out = ttnn.reshape(out, (1, batch_size, patches_h * patches_w, -1))
        return out


class MochiPatchEmbed:
    def __init__(
        self,
        patch_size,
        in_channels,
        embed_dim,
        mesh_device=None,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mesh_device = mesh_device

        # Conv2d projection weights (unfolded)
        # Weight shape: (kernel_h * kernel_w * in_channels, out_channels)
        conv_in_features = patch_size * patch_size * in_channels
        self.proj_weight = None
        self.proj_bias = None

        # Compute kernel config for linear operations
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def to_cached_state_dict(self, path_prefix):
        cache_dict = {}

        # Cache proj_weight
        proj_weight_path = path_prefix + "proj_weight"
        ttnn.dump_tensor(proj_weight_path, self.proj_weight)
        cache_dict["proj_weight"] = proj_weight_path

        # Cache proj_bias if it exists
        if self.proj_bias is not None:
            proj_bias_path = path_prefix + "proj_bias"
            ttnn.dump_tensor(proj_bias_path, self.proj_bias)
            cache_dict["proj_bias"] = proj_bias_path

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.proj_weight = ttnn.load_tensor(cache_dict["proj_weight"], device=self.mesh_device)
        if "proj_bias" in cache_dict:
            self.proj_bias = ttnn.load_tensor(cache_dict["proj_bias"], device=self.mesh_device)

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Load conv2d projection weights
        conv_weight = state_dict["proj.weight"]
        # Convert from (out_channels, in_channels, kh, kw) to (kh*kw*in_channels, out_channels)
        out_channels, in_c, kh, kw = conv_weight.shape
        conv_weight = conv_weight.permute(2, 3, 1, 0)  # (kh, kw, in_c, out_channels)
        conv_weight = conv_weight.reshape(kh * kw * in_c, out_channels)

        self.proj_weight = bf16_tensor(
            conv_weight,
            device=self.mesh_device,
        )

        if "proj.bias" in state_dict:
            bias = state_dict["proj.bias"].reshape(1, 1, 1, -1)
            self.proj_bias = bf16_tensor(
                bias,
                device=self.mesh_device,
            )

    def __call__(self, latent_1BNI):
        """
        latent_1BNI: (1, batch, T * patches_height * patches_width, patch_size * patch_size * in_channels

        returns:
        latent_1BND: (1, batch, T * patches_height * patches_width, embed_dim)
        """

        # Apply unfolded conv2d projection
        latent_1BND = ttnn.linear(
            latent_1BNI,
            self.proj_weight,
            bias=self.proj_bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        return latent_1BND
