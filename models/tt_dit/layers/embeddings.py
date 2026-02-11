# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import torch

import ttnn

from ..parallel.manager import CCLManager
from ..utils.tensor import bf16_tensor, float32_tensor, unflatten
from .linear import ColParallelLinear, Linear
from .module import Module, Parameter

# TODO: Fuse with linear instead
ACT2CLS = {
    "swish": ttnn.swish,
    "silu": ttnn.silu,
    "mish": ttnn.mish,
    "gelu": ttnn.gelu,
    "gelu_tanh": partial(ttnn.gelu, fast_and_approximate_mode=True),
    "relu": ttnn.relu,
}


# positional Encoding.
class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        cos_first: bool = True,
        max_period: int = 10000,
        downscale_freq_shift: float = 0,
        scale: int = 1,
        mesh_device=None,
        use_fp32: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.cos_first = cos_first
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period
        self.mesh_device = mesh_device
        self.use_fp32 = (
            use_fp32  # BF16 not accurate.Ref PCC ~95% with RMSE ~30% vs FP32: PCC = 99.9999 %, RMSE/σ₁ = 0.2 %
        )
        self.time_proj_factor = self._create_time_proj_factor()

    def _create_time_proj_factor(self) -> ttnn.Tensor:
        assert self.num_channels % 2 == 0
        half_dim = self.num_channels // 2

        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        factor = torch.exp(exponent)

        return (
            ttnn.unsqueeze_to_4D(float32_tensor(factor, device=self.mesh_device))
            if self.use_fp32
            else ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=self.mesh_device))
        )

    def forward(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        # Time projection (sinusoidal embedding)
        assert (timestep.dtype == ttnn.float32) == self.use_fp32, "Input timestep must be float32 if use_fp32 is True"
        emb = self.scale * timestep * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        cat_args = [c, s] if self.cos_first else [s, c]
        timesteps_proj = ttnn.concat(cat_args, dim=-1)
        return (
            ttnn.typecast(timesteps_proj, dtype=ttnn.bfloat16)
            if timesteps_proj.dtype == ttnn.float32
            else timesteps_proj
        )


# Helper classes for SD35Transformer2DModel
class TimestepEmbedding(Module):
    def __init__(self, in_channels, time_embed_dim, mesh_device=None, act_fn="silu"):
        super().__init__()

        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.mesh_device = mesh_device
        self.act_fn = ACT2CLS[act_fn]  # TODO: Fuse with linear instead
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True, mesh_device=mesh_device)
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear_1(x)
        x = self.act_fn(x)  # TODO: Fuse with linear instead
        return self.linear_2(x)


class PixArtAlphaTextProjection(Module):
    def __init__(self, in_features, hidden_size, mesh_device=None, act_fn="silu"):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device
        self.act_fn = ACT2CLS[act_fn]  # TODO: Fuse with linear instead
        self.linear_1 = Linear(in_features, hidden_size, bias=True, mesh_device=mesh_device)
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear_1(x)
        x = self.act_fn(x)
        return self.linear_2(x)


class SD35CombinedTimestepTextProjEmbeddings(Module):
    def __init__(self, embedding_dim, pooled_projection_dim, mesh_device=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.mesh_device = mesh_device

        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, mesh_device=mesh_device)
        self.text_embedder = (
            PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, mesh_device=mesh_device)
            if pooled_projection_dim > 0
            else None
        )

        self.time_proj_factor = self._create_time_proj_factor(256)

    def _create_time_proj_factor(self, num_channels) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent)

        return ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=self.mesh_device))

    def forward(self, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor | None = None) -> ttnn.Tensor:
        # Time projection (sinusoidal embedding)
        emb = timestep * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_emb = self.timestep_embedder(timesteps_proj)

        if self.text_embedder is None:
            return timesteps_emb

        if pooled_projection is None:
            msg = "pooled_projection must be provided when text embedder is enabled"
            raise ValueError(msg)

        text_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + text_emb


class CombinedTimestepGuidanceTextProjEmbeddings(Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        pooled_projection_dim: int,
        mesh_device: ttnn.MeshDevice | None = None,
        with_guidance: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.mesh_device = mesh_device
        self.with_guidance = with_guidance

        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, mesh_device=mesh_device)
        self.guidance_embedder = (
            TimestepEmbedding(256, embedding_dim, mesh_device=mesh_device) if with_guidance else None
        )
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, mesh_device=mesh_device)

        self.time_proj_factor = self._create_time_proj_factor(256)

    def _create_time_proj_factor(self, num_channels) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent)

        return ttnn.unsqueeze_to_4D(bf16_tensor(factor, device=self.mesh_device))

    # In order to avoid calling `unsqueeze` in this function, we expect already unsqueezed rank two
    # `timestep` and `guidance` tensors.
    def forward(
        self,
        *,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor | None = None,
        pooled_projection: ttnn.Tensor,
    ) -> ttnn.Tensor:
        batch_size = pooled_projection.shape[0]

        assert len(pooled_projection.shape) == 2
        assert timestep.shape == [batch_size, 1]
        assert timestep.dtype == ttnn.float32, "timesteps require float32 precision"

        emb = timestep * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_emb = self.timestep_embedder(timesteps_proj)

        text_emb = self.text_embedder(pooled_projection)

        if not self.with_guidance:
            return timesteps_emb + text_emb

        assert guidance is not None
        assert guidance.shape == [batch_size, 1]

        emb = guidance * self.time_proj_factor
        c = ttnn.cos(emb)
        s = ttnn.sin(emb)
        guidances_proj = ttnn.concat([c, s], dim=-1)
        guidance_emb = self.guidance_embedder(guidances_proj)

        return timesteps_emb + guidance_emb + text_emb


class PatchEmbed(Module):
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
        sequence_padding: tuple[int, int] = (0, 0),
        mesh_device=None,
    ):
        super().__init__()

        self.height = height // patch_size
        self.width = width // patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.pos_embed_max_size = pos_embed_max_size
        self.mesh_device = mesh_device
        self.tp_mesh_axis = tp_mesh_axis
        self.sp_mesh_axis = sp_mesh_axis
        self.sequence_padding = sequence_padding

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

        self.proj_weight = Parameter(
            total_shape=[in_channels * patch_size * patch_size, embed_dim],
            mesh_axes=[None, tp_mesh_axis],
            device=mesh_device,
        )
        self.proj_bias = Parameter(
            total_shape=[1, 1, 1, embed_dim], mesh_axes=[None, None, None, tp_mesh_axis], device=mesh_device
        )

        seq_len = self.height * self.width + sequence_padding[0] + sequence_padding[1]
        self.pos_embed = Parameter(
            total_shape=[1, seq_len, embed_dim],
            device=mesh_device,
            mesh_axes=[None, sp_mesh_axis, tp_mesh_axis],
        )

    def _cropped_pos_embed(self, pos_embed_param):
        pos_embed_max_size = math.isqrt(pos_embed_param.shape[1])
        top = (pos_embed_max_size - self.height) // 2
        left = (pos_embed_max_size - self.width) // 2

        spatial_pos_embed = pos_embed_param.reshape([1, pos_embed_max_size, pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + self.height, left : left + self.width, :]
        spatial_pos_embed = spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])

        return torch.nn.functional.pad(spatial_pos_embed, (0, 0, *self.sequence_padding))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
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
            state["pos_embed"] = self._cropped_pos_embed(state.pop("pos_embed"))

    def forward(self, latent: ttnn.Tensor, *, already_unfolded: bool = False) -> ttnn.Tensor:
        """
        Forward pass: apply patch projection and add position embeddings.

        Args:
            latent: Input tensor of shape (batch_size, height, width, channels) in NHWC format
                fractured dim 2 along sp_axis

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
                fractured dim 2 along sp_axis, dim 3 along tp_axis
        """
        if already_unfolded:
            latent = ttnn.linear(
                latent,
                self.proj_weight.data,
                bias=self.proj_bias.data,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            latent = self._unfold_conv2d(latent)

        return latent + self.pos_embed.data

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
            self.proj_weight.data,
            bias=self.proj_bias.data,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        out = ttnn.reshape(out, (1, batch_size, patches_h * patches_w, -1))
        return out


class MochiPatchEmbed(Module):
    def __init__(
        self,
        patch_size,
        in_channels,
        embed_dim,
        mesh_device=None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mesh_device = mesh_device

        # Compute kernel config for linear operations
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Conv2d projection weights (unfolded)
        self.proj_weight = Parameter(total_shape=[in_channels * patch_size * patch_size, embed_dim], device=mesh_device)
        self.proj_bias = Parameter(total_shape=[1, 1, 1, embed_dim], device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        conv_weight = state.pop("proj.weight", None)
        if conv_weight is not None:
            # Convert from (out_channels, in_channels, kh, kw) to (kh*kw*in_channels, out_channels)
            out_channels, in_c, kh, kw = conv_weight.shape
            conv_weight = conv_weight.permute(2, 3, 1, 0)  # (kh, kw, in_c, out_channels)
            conv_weight = conv_weight.reshape(kh * kw * in_c, out_channels)
            state["proj_weight"] = conv_weight

        conv_bias = state.pop("proj.bias", None)
        if conv_bias is not None:
            state["proj_bias"] = conv_bias.reshape([1, 1, 1, -1])

    def forward(self, latent_1BNI: ttnn.Tensor) -> ttnn.Tensor:
        """
        latent_1BNI: (1, batch, T * patches_height * patches_width, patch_size * patch_size * in_channels

        returns:
        latent_1BND: (1, batch, T * patches_height * patches_width, embed_dim)
        """

        # Apply unfolded conv2d projection
        latent_1BND = ttnn.linear(
            latent_1BNI,
            self.proj_weight.data,
            bias=self.proj_bias.data,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        return latent_1BND


class WanPatchEmbed(Module):
    def __init__(
        self,
        patch_size,
        in_channels,
        embed_dim,
        mesh_device=None,
        init=False,
        tp_mesh_axis=None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mesh_device = mesh_device
        # Optionally output tensor parallel
        self.tp_mesh_axis = tp_mesh_axis

        assert not init, "WanPatchEmbed does not support initialization"

        # Compute kernel config for linear operations
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.proj_weight = Parameter(
            total_shape=[in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_dim],
            device=mesh_device,
            mesh_axes=[None, self.tp_mesh_axis],
        )
        self.proj_bias = Parameter(
            total_shape=[1, embed_dim],
            device=mesh_device,
            mesh_axes=[None, self.tp_mesh_axis],
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        conv_weight = state.pop("weight", None)
        if conv_weight is not None:
            out_channels, in_c, kt, kh, kw = conv_weight.shape
            assert out_channels == self.embed_dim, "Output channels do not match embed_dim"
            assert in_c == self.in_channels, "Input channels do not match in_channels"
            assert kt == self.patch_size[0], "Patch size does not match patch_size[0]"
            assert kh == self.patch_size[1], "Patch size does not match patch_size[1]"
            assert kw == self.patch_size[2], "Patch size does not match patch_size[2]"

            conv_weight = conv_weight.permute(2, 3, 4, 1, 0)  # (kt, kh, kw, in_c, out_channels)
            conv_weight = conv_weight.reshape(kt * kh * kw * in_c, out_channels)

            state["proj_weight"] = conv_weight

        conv_weight = state.pop("bias", None)
        if conv_weight is not None:
            state["proj_bias"] = conv_weight.reshape(1, -1)

    def forward(self, latent_1BNI: ttnn.Tensor) -> ttnn.Tensor:
        """
        latent_1BNI: (1, batch, pt * ph * pw, kt * kh * kw * in_channels

        returns:
        latent_1BND: (1, batch, pt * ph * pw, embed_dim), optionally fractured embed_dim on TP
        """

        # Apply unfolded conv2d projection
        latent_1BND = ttnn.linear(
            latent_1BNI,
            self.proj_weight.data,
            bias=self.proj_bias.data,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        return latent_1BND


class WanTimeTextImageEmbedding(Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        tp_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        super().__init__()
        assert image_embed_dim is None, "WanTimeTextImageEmbedding does not support image embedding"
        self.mesh_device = mesh_device
        self.tp_mesh_axis = tp_mesh_axis
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, mesh_device=self.mesh_device)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim, mesh_device=self.mesh_device
        )
        self.pre_time_proj_activation = ttnn.silu
        self.time_proj = ColParallelLinear(
            dim, time_proj_dim, bias=True, mesh_device=self.mesh_device, mesh_axis=tp_mesh_axis, ccl_manager=ccl_manager
        )  # Output is fractured according to the older behaviour when sharding from torch. See _prepare_torch_state(...)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh", mesh_device=self.mesh_device
        )

    def forward_timestep(self, timestep: ttnn.Tensor, timestep_seq_len: int | None = None) -> ttnn.Tensor:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = unflatten(timestep, 0, (-1, timestep_seq_len))

        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.pre_time_proj_activation(temb))  # TODO: Fuse with linear instead
        return temb, timestep_proj

    def forward_text(self, encoder_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self.text_embedder(encoder_hidden_states)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # prep timestep proj weight for chunking.
        time_proj_weight = state.pop("time_proj.weight", None)
        time_proj_bias = state.pop("time_proj.bias", None)

        # view as expected output shape
        time_proj_weight = time_proj_weight.unflatten(0, (6, -1))
        time_proj_bias = time_proj_bias.unflatten(0, (6, -1))

        # chunk first dimension by number of devices
        time_proj_weight_chunks = torch.chunk(time_proj_weight, self.mesh_device.shape[self.tp_mesh_axis], dim=1)
        time_proj_bias_chunks = torch.chunk(time_proj_bias, self.mesh_device.shape[self.tp_mesh_axis], dim=1)

        # reshape first dimension to ensure weights per device to ensure weights, data and expected output shape are aligned..
        time_proj_weight_chunks = [chunk.reshape(-1, time_proj_weight.shape[-1]) for chunk in time_proj_weight_chunks]
        time_proj_bias_chunks = [chunk.reshape(-1) for chunk in time_proj_bias_chunks]

        # concatenate back to enable sharding to be applied in preparaton for the chunking.
        state["time_proj.weight"] = torch.cat(time_proj_weight_chunks, 0)
        state["time_proj.bias"] = torch.cat(time_proj_bias_chunks, 0)

    def forward(
        self,
        timestep: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        timestep_seq_len: int | None = None,
    ):
        temb, timestep_proj = self.forward_timestep(timestep, timestep_seq_len)
        encoder_hidden_states = self.forward_text(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class Embedding(Module):
    def __init__(
        self, dictionary_size: int, embedding_size: int, *, device: ttnn.MeshDevice, mesh_axis: int | None = None
    ) -> None:
        super().__init__()

        # use row major layout since ttnn.embedding allocates the embedding tensor again otherwise
        self.weight = Parameter(
            total_shape=[dictionary_size, embedding_size],
            mesh_axes=[None, mesh_axis],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def forward(self, x: ttnn.Tensor, /) -> ttnn.Tensor:
        return ttnn.embedding(x, self.weight.data, layout=ttnn.TILE_LAYOUT)
