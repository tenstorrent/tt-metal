# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKLWan

import ttnn
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae import (
    VaeContext,
    VaeConv2d,
    VaeMidBlock,
    VaeNormDescRms,
    VaeResnetBlock,
    VaeRmsNorm,
    VaeUpsampler,
)
from models.tt_dit.parallel.config import Flux2VaeParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


class FiboVaeDecoder(Module):
    """FIBO's Wan 2.2 (residual) VAE decoder without temporal dimension.

    Mirrors ``QwenImageVaeDecoder`` but with two FIBO-specific differences:

    - Supports an asymmetric decoder via ``decoder_base_dim``.
    - Uses ``FiboVaeUpBlock`` (residual variant: ``WanResidualUpBlock`` in diffusers), which
      adds a duplicate-upsample skip (``avg_shortcut``) parallel to the resnets+upsampler path,
      and keeps the first resnet's ``in_channels`` un-halved.
    """

    def __init__(
        self,
        *,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        out_channels: int = 3,
        patch_size: int = 1,
        parallel_config: Flux2VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        tp = parallel_config.tp_parallel
        wp = parallel_config.w_parallel
        hp = parallel_config.h_parallel

        ctx = VaeContext(
            tp_axis=tp.mesh_axis if tp is not None else None,
            h_mesh_axis=hp.mesh_axis if hp is not None else None,
            h_factor=hp.factor if hp is not None else 1,
            w_mesh_axis=wp.mesh_axis if wp is not None else None,
            w_factor=wp.factor if wp is not None else 1,
            device=device,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        dim = decoder_base_dim if decoder_base_dim is not None else base_dim
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        eps = 1e-12

        self.post_quant_conv = Linear(z_dim, z_dim, mesh_device=device)
        self.conv_in = VaeConv2d(z_dim, dims[0], kernel_size=3, padding=1, ctx=ctx)
        self.mid_block = VaeMidBlock(
            num_channels=dims[0],
            norm=VaeNormDescRms(eps=eps),
            ctx=ctx,
        )

        self.up_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            up_block = FiboVaeUpBlock(
                in_channels=in_dim,
                out_channels=out_dim,
                num_layers=num_res_blocks + 1,
                upsample=i != len(dim_mult) - 1,
                norm=VaeNormDescRms(eps=eps),
                ctx=ctx,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = VaeRmsNorm(out_dim, eps=eps, ctx=ctx, activation_fn="silu")
        self.conv_out = VaeConv2d(out_dim, out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx)

        self._ctx = ctx
        self._patch_size = patch_size
        self._image_channels = out_channels // (patch_size * patch_size)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        _convert_state_dict(state)

        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

        rename_substate(state, "decoder", "")

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """Return the decoded image, replicated, in row-major layout.

        The result has shape (B, H, W * C).
        """
        z = self.post_quant_conv.forward(z)
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            z = self._ctx.ccl_manager.all_gather(z, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        z = self.conv_out.forward(z)
        z = ttnn.clamp(z, min=-1.0, max=1.0)

        # In tile layout the unpatchify's trailing (P, P) dims would be padded to a tile.
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        if self._patch_size > 1:
            z = _unpatchify(z, patch_size=self._patch_size, out_channels=self._image_channels)

        #  A row-major page holds one row of the last dimension, and pages of only C elements make
        # the gather and the read to the host many times slower.
        b, h, w, c = z.shape
        z = ttnn.reshape(z, [b, h, w * c])

        ctx = self._ctx
        if ctx.h_factor > 1:
            z = ctx.ccl_manager.all_gather(z, dim=1, mesh_axis=ctx.h_mesh_axis, use_hyperparams=True)
        if ctx.w_factor > 1:
            z = ctx.ccl_manager.all_gather(z, dim=2, mesh_axis=ctx.w_mesh_axis, use_hyperparams=True)
        return z


class FiboVaeUpBlock(Module):
    """Wan 2.2's residual up block (``WanResidualUpBlock``), 2D-only.

    Same shape as ``VaeUpBlock`` (a stack of ``VaeResnetBlock`` plus an optional upsampler), with
    one extra path: when upsampling, a ``FiboDupUp2D`` duplicate-upsamples the *input* and is
    added to the upsampler's output. The first resnet's ``in_channels`` is also un-halved
    relative to ``VaeUpBlock``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        upsample: bool,
        norm: VaeNormDescRms,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm=norm,
                ctx=ctx,
            )
            for i in range(num_layers)
        )

        if upsample:
            self.upsampler = VaeUpsampler(in_channels=out_channels, out_channels=out_channels, ctx=ctx)
            self.avg_shortcut = FiboDupUp2D(in_channels=in_channels, out_channels=out_channels, factor=2)
        else:
            self.upsampler = None
            self.avg_shortcut = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_in = x

        for block in self.resnets:
            x = block.forward(x)

        if self.upsampler is not None:
            x = self.upsampler.forward(x)
            x = x + self.avg_shortcut.forward(x_in)

        return x


class FiboDupUp2D(Module):
    """``DupUp3D`` from diffusers, restricted to 2D (no temporal upsample).

    Duplicate-upsamples a ``(B, H, W, C_in)`` tensor to ``(B, F·H, F·W, C_out)`` (where ``F`` is
    ``factor_s``) by repeating each input channel ``C_out * F² // C_in`` times along the channel
    axis and then pixel-shuffling F×F. No learnable parameters.

    Only the two repeat counts FIBO's decoder uses are supported: F², a plain nearest-neighbour
    upsample, and F, where ``out[F·h + i, F·w + j, c] = x[h, w, F·c + i]``.
    """

    def __init__(self, *, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()

        repeats, remainder = divmod(out_channels * factor * factor, in_channels)
        if remainder != 0 or repeats not in (factor, factor * factor):
            msg = (
                f"unsupported channel ratio: in_channels {in_channels}, out_channels {out_channels}, "
                f"factor {factor}"
            )
            raise ValueError(msg)
        self._repeats = repeats
        self._factor = factor

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        f = self._factor

        if self._repeats == f * f:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.upsample(x, scale_factor=f)
            return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Row phase i picks every f-th channel, column phase j repeats. The channels are split in
        # a non-last dim, since a row-major page holds one row of the last dim and ops moving
        # pages of only f elements are very slow.
        bs, h, w, c = x.shape
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [bs, h, c // f, f, w])
        x = ttnn.permute(x, (0, 1, 3, 2, 4))
        x = ttnn.reshape(x, [bs, h * f, c // f, w])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.upsample(x, scale_factor=[1, f])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _unpatchify(x: ttnn.Tensor, *, patch_size: int, out_channels: int) -> ttnn.Tensor:
    """``(B, H, W, C * P * P) -> (B, P*H, P*W, C)``, matching diffusers ``AutoencoderKLWan``.

    Diffusers packs the channel axis as ``(C, P_w, P_h)`` outer-to-inner (see
    ``autoencoder_kl_wan.py:927`` — ``permute(0, 1, 6, 4, 2, 3, 5)``) and its unpatchify pairs H
    with ``P_h``, W with ``P_w``. In our HWC reshape ``(B, H, W, C, P_w, P_h)``, that means H is
    paired with the *second* inner patch dim and W with the *first* — hence the permute
    ``(0, 1, 5, 2, 4, 3)`` instead of the more familiar tt_dit pattern ``(0, 1, 4, 2, 5, 3)``
    used by ``vae_flux2`` and the transformer unpatchify methods, whose reference checkpoints
    happen to pack the other way around.
    """
    bs, h, w, _ = x.shape
    x = ttnn.reshape(x, [bs, h, w, out_channels, patch_size, patch_size])
    x = ttnn.permute(x, (0, 1, 5, 2, 4, 3))
    return ttnn.reshape(x, [bs, h * patch_size, w * patch_size, out_channels])


def _convert_state_dict(state: dict[str, torch.Tensor]) -> None:
    """Convert FIBO's AutoencoderKLWan state dict to the 2D layout this decoder expects."""
    rename = {
        "decoder.norm_out.gamma": "decoder.conv_norm_out.gamma",
        "decoder.mid_block.attentions.0.proj.weight": "decoder.mid_block.attentions.0.to_out.0.weight",
        "decoder.mid_block.attentions.0.proj.bias": "decoder.mid_block.attentions.0.to_out.0.bias",
    }
    for src, dst in rename.items():
        state[dst] = state.pop(src)

    # ``WanResidualUpBlock`` stores its upsampler as ``upsampler.resample[1]`` (a Sequential) plus
    # an optional ``upsampler.time_conv`` for the temporal-upsample variant. The 2D decoder uses
    # ``upsampler.conv`` and skips temporal, so rename the spatial conv and drop the time conv.
    for key in list(state.keys()):
        if ".upsampler.resample.1." in key:
            state[key.replace(".upsampler.resample.1.", ".upsampler.conv.")] = state.pop(key)
        elif ".upsampler.time_conv." in key:
            del state[key]

    # The 3D causal convs in the FIBO checkpoint have shape (out, in, T_k, H_k, W_k). Slice off the
    # last temporal kernel (matches what diffusers does on T=1 prefill) so we end up with 2D
    # (out, in, H_k, W_k) weights compatible with ``VaeConv2d`` / ``Conv2d``.
    conv3d_keys = [k for k, v in state.items() if k.startswith("decoder.") and v.ndim == 5]
    for key in conv3d_keys:
        state[key] = state[key][:, :, -1, :, :]
    if "post_quant_conv.weight" in state and state["post_quant_conv.weight"].ndim == 5:
        state["post_quant_conv.weight"] = state["post_quant_conv.weight"][:, :, -1, :, :]

    # Mid-block attention QKV is stored fused in the checkpoint.
    if "decoder.mid_block.attentions.0.to_qkv.weight" in state:
        (
            state["decoder.mid_block.attentions.0.to_q.weight"],
            state["decoder.mid_block.attentions.0.to_k.weight"],
            state["decoder.mid_block.attentions.0.to_v.weight"],
        ) = (
            state.pop("decoder.mid_block.attentions.0.to_qkv.weight").squeeze(2, 3).chunk(3)
        )
        (
            state["decoder.mid_block.attentions.0.to_q.bias"],
            state["decoder.mid_block.attentions.0.to_k.bias"],
            state["decoder.mid_block.attentions.0.to_v.bias"],
        ) = state.pop("decoder.mid_block.attentions.0.to_qkv.bias").chunk(3)
    if "decoder.mid_block.attentions.0.to_out.0.weight" in state:
        w = state["decoder.mid_block.attentions.0.to_out.0.weight"]
        if w.ndim == 4:
            state["decoder.mid_block.attentions.0.to_out.0.weight"] = w.squeeze(2, 3)


class FiboVAEDecoderAdapter:
    """Torch-in (NHWC), torch-out (BCHW) VAE decoder for FIBO's Wan 2.2 VAE.

    Reads the FIBO VAE config eagerly so callers can inspect ``self.config``; weights are loaded
    lazily on ``reload_weights()`` so cache hits skip the torch model entirely.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: Flux2VaeParallelConfig,
        ccl_manager: CCLManager,
        use_torch: bool,
    ) -> None:
        self._name = checkpoint_name
        self._parallel_config = parallel_config
        self._ccl_manager = ccl_manager
        self._device = ccl_manager.mesh_device

        # Eager config-only read; full torch weights are only loaded if needed (use_torch=True or
        # cache.load_model cache miss).
        from diffusers import AutoencoderKLWan  # noqa: PLC0415

        hf_config = AutoencoderKLWan.load_config(checkpoint_name, subfolder="vae")
        self._hf_config = hf_config
        self._latents_scaling = 1.0 / torch.tensor(hf_config["latents_std"])
        self._latents_shift = torch.tensor(hf_config["latents_mean"])
        self.z_dim: int = hf_config["z_dim"]
        self.patch_size: int = hf_config.get("patch_size") or 1
        self.spatial_compression_ratio: int = hf_config["scale_factor_spatial"]

        if use_torch:
            self._torch_vae = AutoencoderKLWan.from_pretrained(
                checkpoint_name, subfolder="vae", torch_dtype=torch.float32
            )
            self._decoder = None
            self._tracer = None
            self._tt_latents_std = None
            self._tt_latents_mean = None
        else:
            self._torch_vae = None
            self._decoder = FiboVaeDecoder(
                base_dim=hf_config["base_dim"],
                decoder_base_dim=hf_config.get("decoder_base_dim"),
                z_dim=hf_config["z_dim"],
                dim_mult=hf_config["dim_mult"],
                num_res_blocks=hf_config["num_res_blocks"],
                out_channels=hf_config.get("out_channels", 3),
                patch_size=hf_config.get("patch_size") or 1,
                device=self._device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._rescale_and_decode, device=self._device, clone_prep_inputs=False)
            self._tt_latents_std = tensor.from_torch(torch.tensor(hf_config["latents_std"]), device=self._device)
            self._tt_latents_mean = tensor.from_torch(torch.tensor(hf_config["latents_mean"]), device=self._device)

    def is_loaded(self) -> bool:
        return self._torch_vae is not None or any(p.is_loaded() for p in self._decoder.parameters())

    def reload_weights(self) -> None:
        if self._decoder is None:
            return

        cache.load_model(
            self._decoder,
            get_torch_state_dict=self._load_torch_state_dict,
            model_name=self._name.split("/")[-1],
            subfolder="vae",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            mesh_device=self._device,
        )

    def _load_torch_state_dict(self) -> dict[str, torch.Tensor]:
        torch_vae = AutoencoderKLWan.from_pretrained(self._name, subfolder="vae")
        return torch_vae.state_dict()

    def _rescale_and_decode(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = z * self._tt_latents_std + self._tt_latents_mean
        return self._decoder.forward(z)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, traced: bool) -> torch.Tensor:
        _, h, w, _ = latents.shape

        if self._torch_vae is not None:
            latents = latents / self._latents_scaling + self._latents_shift
            return self._torch_vae.decode(latents.permute(0, 3, 1, 2).unsqueeze(2)).sample[:, :, 0]

        hp = self._parallel_config.h_parallel
        wp = self._parallel_config.w_parallel
        if hp is not None and h % hp.factor != 0:
            msg = f"latent height {h} not divisible by {hp.factor}"
            raise ValueError(msg)
        if wp is not None and w % wp.factor != 0:
            msg = f"latent width {w} not divisible by {wp.factor}"
            raise ValueError(msg)

        tt_latents = tensor.from_torch(
            latents,
            device=self._device,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=[
                None,
                hp.mesh_axis if hp is not None else None,
                wp.mesh_axis if wp is not None else None,
                None,
            ],
        )
        tt_out = self._tracer(tt_latents, traced=traced)
        torch_out = tensor.to_torch(tt_out)
        b, out_h, _ = torch_out.shape
        torch_out = torch_out.reshape(b, out_h, w * self.spatial_compression_ratio, -1)
        return torch_out.permute(0, 3, 1, 2)
