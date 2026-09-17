# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.conv2d import Conv2d
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import GroupNorm
from ...parallel.config import VAEParallelConfig, vae_all_gather
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from diffusers.models.autoencoders import vae as diffusers_vae

    from ...parallel.config import VAEParallelConfig
    from ...parallel.manager import CCLManager


class ResnetBlock(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        norm_core_grid: ttnn.CoreGrid | None = None,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.norm1 = GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
        )
        self.norm2 = GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
            eps=eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
        )
        # Shard whichever of in/out channels is smaller to minimize communication (all_gather on
        # out_mesh_axis moves in_channels bytes; reduce_scatter on in_mesh_axis moves out_channels
        # bytes). Ties (conv2, always in==out) fall to in_mesh_axis, matching vae.py's VaeConv2d.
        conv1_out_is_greater = out_channels > in_channels
        mesh_axis = parallel_config.tensor_parallel.mesh_axis
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            mesh_device=mesh_device,
            in_mesh_axis=mesh_axis if not conv1_out_is_greater else None,
            out_mesh_axis=mesh_axis if conv1_out_is_greater else None,
            ccl_manager=ccl_manager,
            use_barrier=False,
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            mesh_device=mesh_device,
            in_mesh_axis=mesh_axis,
            ccl_manager=ccl_manager,
            use_barrier=False,
        )
        self.conv_shortcut = (
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
                mesh_device=mesh_device,
                in_mesh_axis=mesh_axis if not conv1_out_is_greater else None,
                out_mesh_axis=mesh_axis if conv1_out_is_greater else None,
                ccl_manager=ccl_manager,
                use_barrier=False,
            )
            if in_channels != out_channels
            else None
        )

    # TODO: Update to use defined members within the class for portability
    @classmethod
    def from_torch(
        cls,
        torch_ref,
        mesh_device=None,
        norm_core_grid=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        resnet_block = cls(
            in_channels=torch_ref.in_channels,
            out_channels=torch_ref.out_channels,
            num_groups=torch_ref.num_groups,
            eps=torch_ref.eps,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        return resnet_block

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x
        x = self.norm1(x)
        x = ttnn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = ttnn.silu(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # Following binary op requires tile layout
        return x + residual


class Upsample2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            use_barrier=False,
        )

    # Fix to align with constructor
    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, mesh_axis=None, parallel_manager=None):
        layer = cls(
            in_channels=torch_ref.in_channels,
            out_channels=torch_ref.out_channels,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            parallel_manager=parallel_manager,
        )
        return layer

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Upsample requires row major.
        x = ttnn.upsample(x, scale_factor=2)
        x = self.conv(x)
        return x


class UpDecoderBlock2D(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_groups: int,
        add_upsample: bool,
        mesh_device: ttnn.MeshDevice,
        norm_core_grid: ttnn.CoreGrid | None = None,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            ResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_groups=resnet_groups,
                eps=1e-6,
                mesh_device=mesh_device,
                norm_core_grid=norm_core_grid,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for i, _ in enumerate(range(num_layers))
        )
        self.upsamplers = ModuleList(
            [Upsample2D(out_channels, out_channels, mesh_device, parallel_config, ccl_manager)] if add_upsample else []
        )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, norm_core_grid=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            in_channels=torch_ref.in_channels,
            out_channels=torch_ref.out_channels,
            num_layers=len(torch_ref.resnets),
            resnet_groups=torch_ref.resnet_groups,
            add_upsample=len(torch_ref.upsamplers) > 0,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


# TODO: Add support for coll and row parallel linear. Fuse qkv computation
class Attention(Module):
    # SDPA chunk sizes keyed by (is_blackhole, tp_factor). Empty by default; callers populate
    # per-config tuning. Falls back to default_sdpa_chunk_size, matching vae.py's VaeAttention.
    sdpa_chunk_size_map: dict[tuple, tuple[int, int]] = {}
    default_sdpa_chunk_size: tuple[int, int] = (128, 128)

    def __init__(
        self,
        *,
        query_dim: int,
        head_dim: int,
        num_heads: int,
        norm_num_groups: int,
        mesh_device: ttnn.MeshDevice,
        norm_core_grid: ttnn.CoreGrid | None = None,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.head_dim * self.num_heads
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.to_q = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_k = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_v = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_out = ModuleList(
            [
                ColParallelLinear(
                    in_features=self.inner_dim,
                    out_features=self.query_dim,
                    mesh_device=mesh_device,
                    mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                )
            ]
        )
        self.group_norm = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.query_dim,
            eps=1e-6,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
        )

        tp_factor = parallel_config.tensor_parallel.factor
        resolved_q_chunk, resolved_k_chunk = self.sdpa_chunk_size_map.get(
            (is_blackhole(), tp_factor),
            self.default_sdpa_chunk_size,
        )
        grid_size = mesh_device.compute_with_storage_grid_size()
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=resolved_q_chunk,
            k_chunk_size=resolved_k_chunk,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self._mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, norm_core_grid=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            query_dim=torch_ref.query_dim,
            head_dim=torch_ref.head_dim,
            num_heads=torch_ref.num_heads,
            norm_num_groups=torch_ref.norm_num_groups,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    @staticmethod
    def reorder_for_attention(x, batch_size, n_heads, head_dim):
        return ttnn.permute(ttnn.reshape(x, (batch_size, -1, n_heads, head_dim)), (0, 2, 1, 3))

    # TODO: Standardize this usage
    def gather_if_sharded(self, x):
        if x.shape[3] < self.to_q.in_features:
            x = vae_all_gather(self.ccl_manager, x, self.parallel_config.tensor_parallel.mesh_axis, use_barrier=False)
        return x

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert len(x.shape) == 4
        residual = x
        # elementwise required to be tilized
        in_layout = x.layout
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        [b, h, w, c] = list(x.shape)

        # No need to transpose like reference. x is already channel last
        x = self.group_norm(x)
        x = self.gather_if_sharded(x)

        # output will be bxhxwx(num_heads*head_dims)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        inner_dim = k.shape[-1]
        head_dim = inner_dim // self.num_heads

        q = self.reorder_for_attention(q, b, self.num_heads, head_dim)
        k = self.reorder_for_attention(k, b, self.num_heads, head_dim)
        v = self.reorder_for_attention(v, b, self.num_heads, head_dim)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (b, h, w, inner_dim))

        for to_out in self.to_out:
            x = to_out(x, compute_kernel_config=self._mm_compute_kernel_config)

        x = x + residual

        x = ttnn.to_layout(x, in_layout)
        return x


class UnetMidBlock2D(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        resnet_groups: int,
        attention_head_dim: int,
        mesh_device: ttnn.MeshDevice,
        norm_core_grid: ttnn.CoreGrid | None = None,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.attentions = ModuleList(
            [
                Attention(
                    query_dim=in_channels,
                    head_dim=attention_head_dim,
                    num_heads=in_channels // attention_head_dim,
                    norm_num_groups=resnet_groups,
                    mesh_device=mesh_device,
                    norm_core_grid=norm_core_grid,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
            ]
        )
        self.resnets = ModuleList(
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=resnet_groups,
                eps=1e-6,
                mesh_device=mesh_device,
                norm_core_grid=norm_core_grid,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(2)
        )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, norm_core_grid=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            in_channels=torch_ref.in_channels,
            resnet_groups=torch_ref.resnet_groups,
            attention_head_dim=torch_ref.attention_head_dim,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


# TODO: Clean up, and factor out duplicate code
class VAEDecoder(Module):
    def __init__(
        self,
        *,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        in_channels: int = 16,
        out_channels: int = 3,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        """
        Initialize the VAEDecoder.
        Args:
            block_out_channels: The number of channels for the updecoder blocks. They are also used to support other layers and blocks
            in_channels: The number of channels in the input image.
            out_channels: The number of channels in the output image.
            layers_per_block: The number of Resnet layers (blocks) in each updecoder.
            norm_num_groups: The number of groups in the normalization layer.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        super().__init__()

        # NOTE: tried overriding GroupNorm's default 8x8 core grid to the full 11x10 Blackhole
        # grid here; group_norm's valid grid is shape-dependent (virtual-row/col constraints
        # tied to Ht/W/num_groups per call site) and 11x10 is invalid for at least one of the
        # VAE's many differently-shaped GroupNorm calls (TT_THROW confirmed on real hardware:
        # "largest valid grid that fits is (x=8,y=8)" for one call). Leaving core_grid=None
        # (the existing per-class default) until a proper per-shape grid sweep is done.
        norm_core_grid = None

        self.conv_in = Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=(3, 3),
            padding=(1, 1),
            mesh_device=mesh_device,
            out_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            use_barrier=False,
        )
        self.mid_block = UnetMidBlock2D(
            in_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.up_blocks = ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        prev_output_channel = reversed_block_out_channels[0]
        for i, output_channel in enumerate(reversed_block_out_channels):
            is_final_block = i == len(reversed_block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_groups=norm_num_groups,
                mesh_device=mesh_device,
                norm_core_grid=norm_core_grid,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=block_out_channels[0],
            eps=1e-6,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
        )

        self.conv_out = Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self._tp_axis = parallel_config.tensor_parallel.mesh_axis
        self._ccl_manager = ccl_manager

    @classmethod
    def from_torch(
        cls,
        torch_ref: diffusers_vae.Decoder,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> VAEDecoder:
        model = cls(
            block_out_channels=[block.resnets[0].conv2.out_channels for block in torch_ref.up_blocks][::-1],
            in_channels=torch_ref.conv_in.in_channels,
            out_channels=torch_ref.conv_out.out_channels,
            layers_per_block=torch_ref.layers_per_block,
            norm_num_groups=torch_ref.mid_block.resnets[0].norm1.num_groups,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        model.load_torch_state_dict(torch_ref.state_dict())
        return model

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = ttnn.silu(x)
        x = vae_all_gather(self._ccl_manager, x, cluster_axis=self._tp_axis, use_barrier=False)
        x = self.conv_out(x)
        return x


class VAEDecoderAdapter:
    """Torch-in (NHWC), torch-out (BCHW) VAE decoder for the SD3.5-family VAE.

    Applies an optional pre-decode unpack, then scaling/shift inversion. Supports both the
    PyTorch and TT-NN implementations; the TT-NN backend supports tracing and dynamic load /
    unload of weights via ``deallocate_weights``/``reload_weights``.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
        use_torch: bool,
        skip_shift: bool = False,
        unpack_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        torch_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(torch_vae, AutoencoderKL)

        self.device = ccl_manager.mesh_device
        self.scaling_factor = torch_vae.config["scaling_factor"]
        self.shift_factor = 0.0 if skip_shift else torch_vae.config["shift_factor"]
        self._unpack_fn = unpack_fn

        if use_torch:
            self._torch_vae = torch_vae
            self._decoder = None
            self._tracer = None
            self._decoder_state_dict = None
        else:
            self._torch_vae = None
            self._decoder = VAEDecoder.from_torch(
                torch_vae.decoder,
                mesh_device=self.device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._decoder.forward, device=self.device, prep_run=False)
            self._decoder_state_dict = torch_vae.decoder.state_dict()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, traced: bool) -> torch.Tensor:
        if self._unpack_fn is not None:
            latents = self._unpack_fn(latents)

        latents = latents / self.scaling_factor + self.shift_factor

        if self._torch_vae is not None:
            return self._torch_vae.decode(latents.permute(0, 3, 1, 2).float()).sample

        tt_latents = tensor.from_torch(latents, device=self.device)
        forward = self._tracer if traced else self._decoder.forward
        tt_out = forward(tt_latents)
        return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)

    def is_loaded(self) -> bool:
        return self._torch_vae is not None or self._decoder.is_loaded()

    def deallocate_weights(self) -> None:
        if self._decoder is not None:
            self._decoder.deallocate_weights()

    def reload_weights(self) -> None:
        if self._decoder is None or self._decoder.is_loaded():
            return
        self._decoder.load_torch_state_dict(self._decoder_state_dict)
