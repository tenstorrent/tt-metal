# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SD3.5 VAE decoder on the shared spatial-parallel VAE library (``vae.py``).

Same AutoencoderKL decoder as ``vae_sd35.VAEDecoder`` (block_out_channels 128/256/512/512, three
resnets per up block, GroupNorm(32), one mid-block attention) but built from the blocks Flux.2, Wan
and LTX use: the image is split across devices by height and/or width with a halo exchange between
neighbours instead of channel tensor parallelism with a full all-gather before every conv, SiLU is
fused into the distributed GroupNorm, Q/K/V is one matmul, nothing synchronizes the host so the
forward traces, and the latent is unpatchified on device. Weights load from the diffusers
``AutoencoderKL`` state dict.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKL

import ttnn

from ...layers.module import Module, ModuleList
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import _BLOCKINGS as _conv3d_blockings
from ...utils.substate import pop_substate, rename_substate
from ...utils.tensor import fast_device_to_host, float_to_uint8
from ...utils.tracing import traced_function
from .vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDescGroup, VaeUpBlock, _all_gather_hw, _norm, _partition_hw

# SD35_VAE_NOGATHER=1: keep the decoded image spatially sharded on device and stitch the shards on
# the host, instead of all-gathering the full image onto every chip and then DMA-reading all four
# identical copies (4x the PCIe traffic; the readback is what contends when 8 columns decode at once).
_NOGATHER = os.environ.get("SD35_VAE_NOGATHER", "0") == "1"

if TYPE_CHECKING:
    from collections.abc import Sequence

# tp / h / w axes; any two of the three on a 2-D mesh.
SD35VaeParallelConfig = Flux2VaeParallelConfig

# conv3d blockings for the decoder's convs at 1024x1024 with the image split 4 ways on width
# (per-device W 256, halo-padded 258), swept 2026-09-17 on a Blackhole Galaxy device with
# sweep_conv3d_sd35.py (brute-force runner from tests/models/wan2_2). The shared conv path looks
# blockings up with H = W = 0, so one entry per (in, out) channel pair serves every resolution the
# pair appears at; where a pair spans resolutions the entry minimizes the summed per-decode time
# (within 4% of the per-resolution optimum). The channel-only fallbacks these replace were 2.5x
# to 3.5x slower (e.g. 128->128 at 1024x258: 2199 -> 638 us; 256->256 upsample conv: 6541 ->
# 1849 us), about 30 ms per decode in total.
_SD35_W4_CONV3D_BLOCKINGS = {
    # (h_factor, w_factor, C_in, C_out, kernel, T, H, W): (C_in_blk, C_out_blk, T_blk, H_blk, W_blk)
    (1, 4, 128, 128, (1, 3, 3), 1, 0, 0): (128, 128, 1, 2, 32),  # up3 resnets @1024x258: 638 us
    (1, 4, 256, 128, (1, 3, 3), 1, 0, 0): (256, 128, 1, 32, 2),  # up3 first resnet @1024x258: 971 us
    (1, 4, 128, 3, (1, 3, 3), 1, 0, 0): (128, 32, 1, 2, 32),  # conv_out @1024x258: 556 us
    (1, 4, 256, 256, (1, 3, 3), 1, 0, 0): (
        256,
        128,
        1,
        4,
        16,
    ),  # up2 resnets @512x130 (544) + upsample @1024x258 (1888)
    (1, 4, 512, 256, (1, 3, 3), 1, 0, 0): (256, 128, 1, 8, 8),  # up2 first resnet @512x130: 1125 us
    (1, 4, 512, 512, (1, 3, 3), 1, 0, 0): (256, 128, 1, 8, 8),  # mid/up0/up1 resnets + upsamples: 209/582/643/2018 us
}
_conv3d_blockings.update(_SD35_W4_CONV3D_BLOCKINGS)


class SD35VaeDecoder(Module):
    def __init__(
        self,
        *,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        z_channels: int = 16,
        norm_num_groups: int = 32,
        parallel_config: SD35VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        use_conv3d: bool = True,
    ) -> None:
        super().__init__()
        ctx = VaeContext(
            tp_axis=parallel_config.tp_parallel.mesh_axis if parallel_config.tp_parallel is not None else None,
            device=device,
            ccl_manager=ccl_manager,
            h_mesh_axis=parallel_config.h_parallel.mesh_axis if parallel_config.h_parallel is not None else None,
            h_factor=parallel_config.h_parallel.factor if parallel_config.h_parallel is not None else 1,
            w_mesh_axis=parallel_config.w_parallel.mesh_axis if parallel_config.w_parallel is not None else None,
            w_factor=parallel_config.w_parallel.factor if parallel_config.w_parallel is not None else 1,
            use_conv3d=use_conv3d,
            # bf16 activations: HiFi2 matches HiFi4 numerically on this decoder (PCC 0.99993 vs
            # 0.99994 against torch) and is 6% faster.
            conv_math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        if (ctx.tp_axis is not None or ctx.h_factor > 1 or ctx.w_factor > 1) and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided when the decoder is parallel"
            raise ValueError(msg)

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]
        eps = 1e-6
        norm = VaeNormDescGroup(num_groups=norm_num_groups, eps=eps)

        # The conv3d weight prep pads input channels to 32; the 16-channel latent conv takes the
        # conv2d path instead (it is a tiny layer either way).
        conv_in_ctx = replace(ctx, use_conv3d=False) if z_channels % 32 != 0 else ctx
        self.conv_in = VaeConv2d(z_channels, channel_counts[0], kernel_size=3, padding=1, ctx=conv_in_ctx)
        self.mid_block = VaeMidBlock(num_channels=channel_counts[0], norm=norm, ctx=ctx)
        self.up_blocks = ModuleList(
            VaeUpBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                upsample=i != len(channel_counts) - 2,
                num_layers=layers_per_block + 1,
                norm=norm,
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )
        self.conv_norm_out = _norm(norm, num_channels=channel_counts[-1], ctx=ctx, activation_fn="silu")
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )
        self._z_channels = z_channels
        self._ctx = ctx

    @property
    def ctx(self) -> VaeContext:
        return self._ctx

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Accepts the full AutoencoderKL state dict or just the decoder's.
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")
        pop_substate(state, "post_quant_conv")
        rename_substate(state, "decoder", "")

    def preprocess_and_unpatchify(
        self,
        z: ttnn.Tensor,
        *,
        height: int,
        width: int,
        patch_size: int,
        scaling_factor: float,
        shift_factor: float,
    ) -> ttnn.Tensor:
        """(1, B, (H/p)*(W/p), p*p*C) patchified DiT latent, replicated -> (B, H/h, W/w, C) sharded.

        Inverts the scaling the pipeline applied (z / scaling_factor + shift_factor) and undoes
        the SD3 patchify order (token = (h, w), channel = (p_h, p_w, C)) on device.
        """
        p = patch_size
        assert height % p == 0 and width % p == 0, f"latent {height}x{width} must be divisible by patch {p}"
        one, batch, n_tokens, feat = z.shape
        assert (
            one == 1 and n_tokens == (height // p) * (width // p) and feat == p * p * self._z_channels
        ), f"unexpected latent shape {tuple(z.shape)} for {height}x{width}, patch {p}, C {self._z_channels}"
        if z.layout != ttnn.TILE_LAYOUT:
            z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        z = ttnn.multiply(z, 1.0 / scaling_factor)
        z = ttnn.add(z, shift_factor)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, [batch, height // p, width // p, p, p, self._z_channels])
        z = ttnn.permute(z, [0, 1, 3, 2, 4, 5])
        z = ttnn.reshape(z, [batch, height, width, self._z_channels])
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        return _partition_hw(self._ctx, z)

    def _forward_body(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = self.conv_in.forward(z)
        z = self.mid_block.forward(z)
        for block in self.up_blocks:
            z = block.forward(z)
        z = self.conv_norm_out.forward(z)
        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            z = self._ctx.ccl_manager.all_gather(z, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)
        z = self.conv_out.forward(z)
        if _NOGATHER:
            return z
        return _all_gather_hw(self._ctx, z)

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def forward(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        return self._forward_body(z)

    # SD35_VAE_FOLD=1: one trace for un-scale/unpatchify + decoder + float->uint8, so the VAE
    # phase is a single trace launch plus one DMA read instead of ~12 untraced dispatches whose
    # host round-trip latency is exposed when several columns decode at the same moment.
    _fold_args: dict | None = None

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def decode_fused(self, tt_latents: ttnn.Tensor, /) -> ttnn.Tensor:
        z = self.preprocess_and_unpatchify(tt_latents, **self._fold_args)
        return float_to_uint8(self._forward_body(z))


def sd35_vae_parallel_config(mesh_device: ttnn.MeshDevice) -> SD35VaeParallelConfig:
    """Spatial split over every mesh axis with more than one device: width on the first such axis,
    height on the second (2x2). Width beat height by 15% on a 4x1 Blackhole column."""
    axes = [axis for axis in (0, 1) if mesh_device.shape[axis] > 1]
    if not axes:
        return SD35VaeParallelConfig()
    if len(axes) == 1:
        return SD35VaeParallelConfig.from_axes(mesh_device, w_axis=axes[0])
    return SD35VaeParallelConfig.from_axes(mesh_device, w_axis=axes[0], h_axis=axes[1])


class SD35SpatialVaeAdapter:
    """Device-in, uint8-host-out decode for the SD3.5 pipeline.

    Takes the DiT's patchified latent as a replicated device tensor (1, B, N, p*p*C), unpatchifies
    and shards it on device, runs the spatial-parallel decoder (traced or not) and reads back the
    image as uint8 (B, H, W, 3).
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: SD35VaeParallelConfig | None = None,
        use_conv3d: bool = True,
        patch_size: int = 2,
    ) -> None:
        torch_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(torch_vae, AutoencoderKL)
        cfg = torch_vae.config
        assert not cfg.get("use_post_quant_conv", False), "SD3.5-style VAE without post_quant_conv expected"
        self.scaling_factor = float(cfg["scaling_factor"])
        self.shift_factor = float(cfg["shift_factor"] or 0.0)
        self.patch_size = patch_size
        self.device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config if parallel_config is not None else sd35_vae_parallel_config(mesh_device)
        self.decoder = SD35VaeDecoder(
            out_channels=cfg["out_channels"],
            block_out_channels=tuple(cfg["block_out_channels"]),
            layers_per_block=cfg["layers_per_block"],
            z_channels=cfg["latent_channels"],
            norm_num_groups=cfg["norm_num_groups"],
            parallel_config=self.parallel_config,
            device=mesh_device,
            ccl_manager=ccl_manager,
            use_conv3d=use_conv3d,
        )
        self.decoder.load_torch_state_dict(torch_vae.decoder.state_dict())
        del torch_vae

    def decode_device(self, tt_latents: ttnn.Tensor, *, height: int, width: int, traced: bool) -> torch.Tensor:
        """tt_latents: (1, B, N, p*p*C) replicated; height/width in latent pixels. Returns uint8 (B, H, W, 3)."""
        z = self.decoder.preprocess_and_unpatchify(
            tt_latents,
            height=height,
            width=width,
            patch_size=self.patch_size,
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
        )
        image = self.decoder.forward(z, traced=traced)
        return fast_device_to_host(
            image, self.device, self._concat_dims(), ccl_manager=self.ccl_manager, pre_transfer_fn=float_to_uint8
        )

    def _concat_dims(self) -> list[int | None]:
        """Host concat dims for fast_device_to_host: none when the image is replicated, else the
        (B, H, W, C) dims the decoder shards over each mesh axis."""
        if not _NOGATHER:
            return [None, None]
        ctx = self.decoder._ctx
        dims: list[int | None] = [None, None]
        if ctx.h_factor > 1:
            dims[ctx.h_mesh_axis] = 1
        if ctx.w_factor > 1:
            dims[ctx.w_mesh_axis] = 2
        return dims

    def decode_device_fused(self, tt_latents: ttnn.Tensor, *, height: int, width: int, traced: bool) -> torch.Tensor:
        """Same result as decode_device, with preprocessing and uint8 conversion inside the VAE trace."""
        self.decoder._fold_args = dict(
            height=height,
            width=width,
            patch_size=self.patch_size,
            scaling_factor=self.scaling_factor,
            shift_factor=self.shift_factor,
        )
        image_u8 = self.decoder.decode_fused(tt_latents, traced=traced)
        return fast_device_to_host(image_u8, self.device, self._concat_dims(), ccl_manager=self.ccl_manager)
