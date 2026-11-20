# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX 3 Klein VAE encoder on fixed reference canvases. A prepared reference (sides multiples of 16, at most 4 MP)
sits in the top-left corner of the smallest canvas that holds it and the rest is masked out at runtime, so every
reference shape runs the programs of one of eight canvases. The masked encoder matches the encoder on the image alone:
the area right of and below the image is zeroed before every 3x3 conv and strided downsample (their zero padding),
GroupNorm statistics cover the image only, and the mid-block attention biases the keys outside it away."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from ...layers.module import Module, Parameter
from ...utils import tensor
from .vae import VaeAttention, VaeResnetBlock
from .vae_flux2 import Flux2VaeEncoder

# Canvases (width, height), smallest first; every prepared reference fits one. 2816x1536 / 1536x2816 hold the 16:9 /
# 9:16 references up to 4 MP at 1.04x their area; the 7.3 MP pair the wider ones. Sides are multiples of 256, so every
# level down to the latent (/ 8) is whole tiles.
REFERENCE_CANVASES = (
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
    (2816, 1536),
    (1536, 2816),
    (2304, 2304),
    (4096, 1792),
    (1792, 4096),
)
LEVELS = 4
KEY_MASK_BIAS = -1.0e5
GROUP_NORM_EPS = 1e-6
GROUPS = 32


def canvas_for(width: int, height: int) -> tuple[int, int]:
    for w, h in REFERENCE_CANVASES:
        if width <= w and height <= h:
            return w, h
    raise ValueError(f"no reference canvas holds {width}x{height}")


@dataclass
class CanvasLevelMask:
    rows: ttnn.Tensor  # [1, H, 1, 1] fp32, 1 inside the image
    rows16: ttnn.Tensor
    cols16: ttnn.Tensor  # [1, 1, W, 1]
    inv_n: ttnn.Tensor  # [1, 1, 1, 1] fp32: 1 / image pixels at this level

    def apply(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Zeroes bf16 `x` outside the image, in place."""
        ttnn.multiply_(x, self.cols16)
        return ttnn.multiply_(x, self.rows16)

    def deallocate(self) -> None:
        for t in (self.rows, self.rows16, self.cols16, self.inv_n):
            ttnn.deallocate(t)


class CanvasMasks:
    """The masks of one canvas for an image of (width, height) pixels in its top-left corner: each level's row and
    column masks and the mid-block attention's key bias."""

    def __init__(self, device: ttnn.MeshDevice, *, canvas: tuple[int, int], image: tuple[int, int]) -> None:
        (cw, ch), (iw, ih) = canvas, image
        self._levels: dict[tuple[int, int], CanvasLevelMask] = {}
        for level in range(LEVELS):
            w, h, vw, vh = cw >> level, ch >> level, iw >> level, ih >> level
            rows = (torch.arange(h) < vh).float().reshape(1, h, 1, 1)
            cols = (torch.arange(w) < vw).float().reshape(1, 1, w, 1)
            upload = [
                tensor.from_torch(t, device=device, dtype=dtype, mesh_axes=[None] * 4)
                for t, dtype in (
                    (rows, ttnn.float32),
                    (rows, ttnn.bfloat16),
                    (cols, ttnn.bfloat16),
                )
            ]
            inv_n = tensor.from_torch(
                torch.full((1, 1, 1, 1), 1.0 / (vw * vh)), device=device, dtype=ttnn.float32, mesh_axes=[None] * 4
            )
            self._levels[(h, w)] = CanvasLevelMask(*upload, inv_n)
        # the attention keys' bias in 32 columns, which a query extension of 1 / 32 per column sums back to the bias:
        # (inside - 1) * 1e5 is exactly 0 inside (no bf16 1e5 - 1e5 cancellation)
        inside = ((torch.arange(h) < vh)[:, None] & (torch.arange(w) < vw)[None, :]).float()
        bias = (inside - 1.0).reshape(1, 1, h * w, 1).expand(1, 1, h * w, 32) * -KEY_MASK_BIAS
        self.key_bias = tensor.from_torch(bias.contiguous(), device=device, mesh_axes=[None] * 4)

    def __call__(self, x: ttnn.Tensor) -> CanvasLevelMask:
        return self._levels[(int(x.shape[1]), int(x.shape[2]))]

    def deallocate(self) -> None:
        for mask in self._levels.values():
            mask.deallocate()
        ttnn.deallocate(self.key_bias)


class MaskedGroupNorm(Module):
    """GroupNorm (+ SiLU) over the image area of a canvas: two-pass statistics of the masked bf16 activation, summed in
    fp32, the output zeroed outside the image. Channels split over the TP axis (whole groups)."""

    def __init__(self, num_channels: int, *, ctx, silu: bool) -> None:
        super().__init__()
        tp = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        self.num_channels = num_channels
        self._local = num_channels // tp
        self._silu = silu
        shape, axes = [1, 1, 1, num_channels], [None, None, None, ctx.tp_axis]
        self.weight = Parameter(total_shape=shape, device=ctx.device, dtype=ttnn.float32, mesh_axes=axes)
        self.bias = Parameter(total_shape=shape, device=ctx.device, dtype=ttnn.float32, mesh_axes=axes)
        # per-channel sums -> per-channel group means: sums @ groups
        self.groups = Parameter(total_shape=[1, 1, self._local, self._local], device=ctx.device, dtype=ttnn.float32)
        self._fp32_config = ttnn.init_device_compute_kernel_config(
            ctx.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("weight", "bias"):
            if name in state:
                state[name] = state[name].float().reshape(1, 1, 1, -1)
        group = torch.arange(self._local) // (self.num_channels // GROUPS)
        state["groups"] = (group[:, None] == group[None, :]).float()[None, None] / (self.num_channels // GROUPS)

    def _group_mean(self, x: ttnn.Tensor, mask: CanvasLevelMask) -> ttnn.Tensor:
        """Group means over the image of `x` (zero right of the image; freed here): row sums rounded to bf16, then the
        image rows summed in fp32."""
        rows = ttnn.sum(x, dim=2, keepdim=True, compute_kernel_config=self._fp32_config)
        ttnn.deallocate(x)
        rows = ttnn.multiply(ttnn.typecast(rows, ttnn.float32), mask.rows)
        sums = ttnn.sum(rows, dim=1, keepdim=True, compute_kernel_config=self._fp32_config)
        return ttnn.multiply(ttnn.matmul(sums, self.groups.data, compute_kernel_config=self._fp32_config), mask.inv_n)

    def forward(self, x: ttnn.Tensor, mask: CanvasLevelMask) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        mean = self._group_mean(ttnn.multiply(x, mask.cols16), mask)
        square = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
        centred = ttnn.subtract(x, ttnn.typecast(mean, ttnn.bfloat16), activations=square)
        var = self._group_mean(ttnn.multiply_(centred, mask.cols16), mask)
        scale = ttnn.multiply(ttnn.rsqrt(ttnn.add(var, GROUP_NORM_EPS)), self.weight.data)
        shift = ttnn.subtract(self.bias.data, ttnn.multiply(mean, scale))
        act = {"activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]} if self._silu else {}
        out = ttnn.multiply(x, ttnn.typecast(scale, ttnn.bfloat16))
        return mask.apply(ttnn.add_(out, ttnn.typecast(shift, ttnn.bfloat16), **act))


class CanvasVaeEncoder(Flux2VaeEncoder):
    """Flux2VaeEncoder with masked GroupNorms and a masked mid-block attention; one instance serves every canvas."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        ctx = self._ctx
        for block in (*self.down_blocks, self.mid_block):
            for r in block.resnets:
                r.norm1 = MaskedGroupNorm(r.conv1.inner.in_channels, ctx=ctx, silu=True)
                r.norm2 = MaskedGroupNorm(r.conv2.inner.in_channels, ctx=ctx, silu=True)
        channels = self.conv_out.inner.in_channels
        for attn in self.mid_block.attentions:
            attn.norm = MaskedGroupNorm(channels, ctx=ctx, silu=False)
        self.conv_norm_out = MaskedGroupNorm(channels, ctx=ctx, silu=True)

    def _resnet(self, r: VaeResnetBlock, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        h = r.conv1.forward(r.norm1.forward(x, masks(x)))
        h = r.conv2.forward(r.norm2.forward(h, masks(h)))
        if r.conv_shortcut is not None:
            x = r.conv_shortcut.forward(x)
        return x + h

    def _attention(self, attn: VaeAttention, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        n, h, w, _ = (int(d) for d in x.shape)
        xn = ttnn.reshape(attn.norm.forward(x, masks(x)), (n, 1, h * w, -1))
        ccl = attn._ccl_manager
        if ccl is not None:
            xn = ccl.all_gather(xn, dim=-1, mesh_axis=attn._tp_axis, use_hyperparams=True)
        q, k, v = attn.to_qkv.forward(xn, compute_kernel_config=attn._mm_compute_kernel_config)
        if ccl is not None:
            q, k, v = (ccl.all_gather(t, dim=-1, mesh_axis=attn._tp_axis, use_hyperparams=True) for t in (q, k, v))
        channels = int(q.shape[-1])
        # one extra tile column: the queries' 1 / 32 per column picks up the keys' bias, the values' zeros add nothing
        q = ttnn.concat([q, ttnn.full_like(masks.key_bias, 1.0 / 32)], dim=-1)
        k = ttnn.concat([k, masks.key_bias], dim=-1)
        v = ttnn.concat([v, ttnn.zeros_like(masks.key_bias)], dim=-1)
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=channels**-0.5,
            program_config=attn._sdpa_program_config,
            compute_kernel_config=attn._sdpa_compute_kernel_config,
        )
        o = ttnn.slice(o, [0, 0, 0, 0], [n, 1, h * w, channels])
        o = attn.to_out.forward(o, compute_kernel_config=attn._mm_compute_kernel_config)
        return ttnn.reshape(o, (n, h, w, -1)) + x

    def forward_canvas(self, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        """The latent mean of the image in the canvas `x` [1, H, W, 3] (zeros outside the image), replicated."""
        x = self.conv_in.forward(x)
        for block in self.down_blocks:
            for r in block.resnets:
                x = self._resnet(r, x, masks)
            if block.downsampler is not None:
                # zero right of and below the image: the strided conv's padding
                x = block.downsampler.forward(masks(x).apply(x))
        mid = self.mid_block
        x = self._resnet(mid.resnets[0], x, masks)
        x = self._attention(mid.attentions[0], x, masks)
        x = self._resnet(mid.resnets[1], x, masks)
        x = self.conv_norm_out.forward(x, masks(x))
        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            x = self._ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)
        return self.conv_out.forward(x)
