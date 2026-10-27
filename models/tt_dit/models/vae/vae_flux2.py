# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate
from ...utils.tracing import traced_function
from .vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDescGroup, VaeRowPad, VaeUpBlock, _all_gather_hw, _norm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L271
class Flux2VaeDecoder(Module):
    _PATCH_SIZE = 2

    def __init__(
        self,
        *,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        z_channels: int = 32,
        parallel_config: Flux2VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        use_conv3d: bool = False,
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
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]
        eps = 1e-6

        self.post_quant_conv = Linear(z_channels, z_channels, mesh_device=ctx.device)
        self.conv_in = VaeConv2d(z_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)

        self.mid_block = VaeMidBlock(
            num_channels=channel_counts[0],
            norm=VaeNormDescGroup(num_groups=32, eps=eps),
            ctx=ctx,
        )

        self.up_blocks = ModuleList(
            VaeUpBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                upsample=i != len(channel_counts) - 2,
                num_layers=layers_per_block + 1,
                norm=VaeNormDescGroup(num_groups=32, eps=eps),
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )

        self.conv_norm_out = _norm(
            VaeNormDescGroup(num_groups=32, eps=1e-6),
            num_channels=channel_counts[-1],
            ctx=ctx,
            activation_fn="silu",
        )
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        bn_size = self._PATCH_SIZE**2 * z_channels
        self.bn_running_mean = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_running_var = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_eps = 1e-4

        self._ctx = ctx
        self._row_pad_masks: dict = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # remove encoder state
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

        if "bn.running_mean" in state:
            state["bn_running_mean"] = state.pop("bn.running_mean")
        if "bn.running_var" in state:
            state["bn_running_var"] = state.pop("bn.running_var")
        state.pop("bn.num_batches_tracked", None)

        rename_substate(state, "decoder", "")

    def _inv_normalize(self, z: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.sqrt(self.bn_running_var.data + self.bn_eps)
        m = self.bn_running_mean.data
        return z * s + m

    def preprocess_and_unpatchify(self, z: ttnn.Tensor, *, height: int, width: int) -> ttnn.Tensor:
        # N, (H / P) * (W / P), C * P * P -> N, H/h_factor, W, C
        # Input may be H-sharded (h_factor rows per device) or fully replicated.
        # W-sharding is applied by the caller after this call, once the spatial layout
        # is restored (H/W are interleaved in patchified tokens so W can't be split here).
        p = self._PATCH_SIZE
        assert (
            height % (p * self._ctx.h_factor) == 0
        ), f"height {height} must be divisible by patch {p} * h_factor {self._ctx.h_factor}"
        assert (
            width % (p * self._ctx.w_factor) == 0
        ), f"width {width} must be divisible by patch {p} * w_factor {self._ctx.w_factor}"
        height_local = height // self._ctx.h_factor

        z = self._inv_normalize(z)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = z.reshape([z.shape[0], height_local // p, width // p, -1, p, p])
        z = ttnn.permute(z, [0, 1, 4, 2, 5, 3])
        z = z.reshape([z.shape[0], height_local, width, -1])
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        return z

    def row_padding(self, *, logical_h: int, padded_h: int) -> VaeRowPad:
        """The ``VaeRowPad`` of the latent level (``z``'s rows: ``padded_h`` over the H axis, the first ``logical_h``
        the image) with the masks of every decoder level built (once per size): a traced capture allocates none."""
        pad = VaeRowPad(self._ctx, logical_h=logical_h, padded_h=padded_h, cache=self._row_pad_masks)
        level = pad
        for _ in range(len(self.up_blocks)):
            if level.active:
                _ = level.mask
            level = level.upsampled()
        return pad

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def forward(self, z: ttnn.Tensor, /, *, logical_h: int | None = None) -> ttnn.Tensor:
        """``logical_h`` (H-sharded decoder only): the latent rows of the image when ``z`` holds more (rows padded at
        the bottom so every level splits into whole tiles per device); the padding rows never reach an image row
        (``VaeRowPad``) and the output is cropped to ``logical_h * 8`` rows."""
        # Input arrives H-sharded when SP is active (preprocess_and_unpatchify uses per-device
        # H), or replicated when SP is off — both cases work for post_quant_conv (per-position
        # Linear) and the SP-aware conv pyramid below.
        pad = None
        if logical_h is not None and z.shape[1] * self._ctx.h_factor > logical_h:
            pad = self.row_padding(logical_h=logical_h, padded_h=z.shape[1] * self._ctx.h_factor)

        z = self.post_quant_conv.forward(z)

        if pad is None:
            z = self.conv_in.forward(z)
            z = self.mid_block.forward(z)

            for block in self.up_blocks:
                z = block.forward(z)

            z = self.conv_norm_out.forward(z)
        else:
            z = self.conv_in.forward(z, logical_h=pad.logical_h)
            z = self.mid_block.forward(z, pad=pad)

            for block in self.up_blocks:
                z = block.forward(z, pad=pad)
                if block.upsampler is not None:
                    pad = pad.upsampled()

            z = self.conv_norm_out.forward(z, pad=pad)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            z = self._ctx.ccl_manager.all_gather(z, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        if pad is None:
            z = self.conv_out.forward(z)
            return _all_gather_hw(self._ctx, z)

        z = self.conv_out.forward(z, logical_h=pad.logical_h)
        z = _all_gather_hw(self._ctx, z)
        return ttnn.slice(z, [0, 0, 0, 0], [z.shape[0], pad.logical_h, z.shape[2], z.shape[3]])
