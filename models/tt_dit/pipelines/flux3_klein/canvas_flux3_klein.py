# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The VAE encoder on fixed reference canvases. A reference (BFL's prepared image, sides multiples of 16) sits in the
top-left corner of the smallest canvas that holds it, the rest masked out at runtime, so every reference size runs the
programs of its canvas. The masked encoder matches the encoder on the image alone: the area right of and below the
image is zeroed before every 3x3 conv and strided downsample (their zero padding), GroupNorm statistics cover the image
only, and the mid-block attention biases the keys outside it away."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from ...layers.module import Module, Parameter
from ...models.vae.encoder_flux2 import Flux2VaeEncoder
from ...models.vae.vae import VaeAttention, VaeResnetBlock, _all_gather_hw, _partition_hw, sdpa_config_for_rows
from ...utils import tensor

# 1:1, 16:9 and 9:16 canvases (width, height) at 1 MP and 4 MP; every legal reference (fitted <= 4 MP, sides <= 4096,
# multiples of 16) fits one. Sides are multiples of 256: rows split evenly over 32 devices through the three
# downsamplers, and every level's width down to the latent's (/ 8) is whole tiles. The 4 MP 16:9 canvases reach the
# 4096 side (panoramas); 2304 is the smallest square that holds every reference the 1792 sides do not.
REFERENCE_CANVASES = ((1024, 1024), (1536, 1024), (1024, 1536), (2304, 2304), (4096, 1792), (1792, 4096))
LEVELS = 4
KEY_MASK_BIAS = -1.0e5
GROUP_NORM_EPS = 1e-6
GROUPS = 32
_CANVASES_BY_AREA = tuple(sorted(REFERENCE_CANVASES, key=lambda s: (s[0] * s[1], s)))


def reference_canvases() -> tuple[tuple[int, int], ...]:
    return _CANVASES_BY_AREA


def canvas_for(width: int, height: int) -> tuple[int, int]:
    for w, h in reference_canvases():
        if width <= w and height <= h:
            return w, h
    raise ValueError(f"no reference canvas holds {width}x{height}")


class MaskedGroupNorm(Module):
    """GroupNorm (+ SiLU) over the image area of a canvas: two-pass fp32 statistics of the masked activation, summed
    over the H shards, the output zeroed outside the image. Channels split over the TP axis (whole groups)."""

    def __init__(self, num_channels: int, *, ctx, silu: bool, gathered: bool = False) -> None:
        super().__init__()
        tp = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        self.num_channels = num_channels
        self._local = num_channels // tp
        self._silu = silu
        self._ctx = ctx
        # gathered: the activation holds the full canvas on every device (no H reduction)
        self._h_axis = None if gathered or ctx.h_factor == 1 else ctx.h_mesh_axis
        shape, axes = [1, 1, 1, num_channels], [None, None, None, ctx.tp_axis]
        self.weight = Parameter(total_shape=shape, device=ctx.device, dtype=ttnn.float32, mesh_axes=axes)
        self.bias = Parameter(total_shape=shape, device=ctx.device, dtype=ttnn.float32, mesh_axes=axes)
        # per-channel group mean of per-channel sums: x @ groups
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

    def _sum(self, t: ttnn.Tensor) -> ttnn.Tensor:
        n, h, w, c = (int(d) for d in t.shape)
        s = ttnn.sum(ttnn.reshape(t, (1, 1, h * w, c)), dim=2, keepdim=True)
        if self._h_axis is not None:
            gathered = self._ctx.ccl_manager.all_gather(s, dim=1, mesh_axis=self._h_axis, use_hyperparams=False)
            ttnn.deallocate(s)
            s = ttnn.sum(gathered, dim=1, keepdim=True)
            ttnn.deallocate(gathered)
        return s

    def _group_mean(self, t: ttnn.Tensor, inv_n: ttnn.Tensor) -> ttnn.Tensor:
        s = self._sum(t)
        g = ttnn.matmul(s, self.groups.data, compute_kernel_config=self._fp32_config)
        ttnn.deallocate(s)
        out = ttnn.multiply(g, inv_n)
        ttnn.deallocate(g)
        return out

    def forward(self, x: ttnn.Tensor, mask: CanvasLevelMask) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x32 = ttnn.typecast(x, ttnn.float32)
        xm = mask.apply(x32)
        mean = self._group_mean(xm, mask.inv_n)
        ttnn.deallocate(xm)
        centred = ttnn.subtract(x32, mean)
        d = mask.apply(centred)
        ttnn.deallocate(centred)
        sq = ttnn.multiply(d, d)
        ttnn.deallocate(d)
        var = self._group_mean(sq, mask.inv_n)
        ttnn.deallocate(sq)
        rstd = ttnn.rsqrt(ttnn.add(var, GROUP_NORM_EPS))
        ttnn.deallocate(var)
        scale = ttnn.multiply(rstd, self.weight.data)
        ttnn.deallocate(rstd)
        shift = ttnn.subtract(self.bias.data, ttnn.multiply(mean, scale))
        ttnn.deallocate(mean)
        scaled = ttnn.multiply(x32, scale)
        ttnn.deallocate(x32)
        act = {"activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]} if self._silu else {}
        y = ttnn.add(scaled, shift, **act)
        ttnn.deallocate(scaled)
        out = mask.apply(y, dtype=ttnn.bfloat16)
        ttnn.deallocate(y)
        return out


@dataclass
class CanvasLevelMask:
    rows: ttnn.Tensor  # [1, H, 1, 1] fp32 (the device's H shard), 1 inside the image
    cols: ttnn.Tensor  # [1, 1, W, 1] fp32
    rows16: ttnn.Tensor
    cols16: ttnn.Tensor
    inv_n: ttnn.Tensor  # [1, 1, 1, 1] fp32: 1 / image pixels at this level

    def apply(self, x: ttnn.Tensor, dtype=None) -> ttnn.Tensor:
        fp32 = x.dtype == ttnn.float32
        t = ttnn.multiply(x, self.cols if fp32 else self.cols16)
        out = ttnn.multiply(t, self.rows if fp32 else self.rows16, **({"dtype": dtype} if dtype is not None else {}))
        ttnn.deallocate(t)
        return out

    def deallocate(self) -> None:
        for t in (self.rows, self.cols, self.rows16, self.cols16, self.inv_n):
            ttnn.deallocate(t)


class CanvasMasks:
    """Per-request masks of one canvas for an image of (width, height) pixels in its top-left corner: each level's
    row / column masks (rows H-sharded, and full for the gathered mid-block), and the attention key bias."""

    def __init__(self, device, *, canvas: tuple[int, int], image: tuple[int, int], h_axis: int | None) -> None:
        cw, ch = canvas
        iw, ih = image
        self.canvas, self.image = canvas, image
        self._shard: dict[tuple[int, int], CanvasLevelMask] = {}
        self._full: dict[tuple[int, int], CanvasLevelMask] = {}
        hf = device.shape[h_axis] if h_axis is not None else 1
        for level in range(LEVELS):
            w, h, vw, vh = cw >> level, ch >> level, iw >> level, ih >> level
            rows = (torch.arange(h) < vh).float().reshape(1, h, 1, 1)
            cols = (torch.arange(w) < vw).float().reshape(1, 1, w, 1)
            inv_n = torch.full((1, 1, 1, 1), 1.0 / (vw * vh))

            def up(t, axes=None, dtypes=(ttnn.float32, ttnn.bfloat16)):
                return [tensor.from_torch(t, device=device, dtype=dt, mesh_axes=axes or [None] * 4) for dt in dtypes]

            r32, r16 = up(rows, [None, h_axis, None, None])
            c32, c16 = up(cols)
            (n32,) = up(inv_n, dtypes=(ttnn.float32,))
            self._shard[(h // hf, w)] = CanvasLevelMask(r32, c32, r16, c16, n32)
            if level == LEVELS - 1:
                fr32, fr16 = up(rows)
                fc32, fc16 = up(cols)
                (fn32,) = up(inv_n, dtypes=(ttnn.float32,))
                self._full[(h, w)] = CanvasLevelMask(fr32, fc32, fr16, fc16, fn32)
                # the attention keys' bias, built on device: 0 inside the image, KEY_MASK_BIAS outside, in 32 columns
                # that a query extension of 1 / 32 per column sums back to the bias
                cols32 = tensor.from_torch(cols.expand(1, 1, w, 32).contiguous(), device=device, mesh_axes=[None] * 4)
                rows32 = ttnn.repeat(fr16, ttnn.Shape((1, 1, w, 32)))
                inside = ttnn.multiply(rows32, cols32)
                for t in (rows32, cols32):
                    ttnn.deallocate(t)
                # (inside - 1) * 1e5: exactly 0 inside (no bf16 1e5 - 1e5 cancellation)
                outside = ttnn.subtract(inside, 1.0)
                ttnn.deallocate(inside)
                bias = ttnn.multiply(outside, -KEY_MASK_BIAS)
                ttnn.deallocate(outside)
                self.key_bias = ttnn.reshape(bias, (1, 1, h * w, 32))

    def shard(self, h: int, w: int) -> CanvasLevelMask:
        return self._shard[(h, w)]

    def full(self, h: int, w: int) -> CanvasLevelMask:
        return self._full[(h, w)]

    def deallocate(self) -> None:
        for m in (*self._shard.values(), *self._full.values()):
            m.deallocate()
        ttnn.deallocate(self.key_bias)


class CanvasVaeEncoder(Flux2VaeEncoder):
    """Flux2VaeEncoder with masked GroupNorms and a masked mid-block attention, for one canvas shape."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        ctx = self._ctx
        for block in (*self.down_blocks, self.mid_block):
            for r in block.resnets:
                r.norm1 = MaskedGroupNorm(r.conv1.inner.in_channels, ctx=ctx, silu=True)
                r.norm2 = MaskedGroupNorm(r.conv2.inner.in_channels, ctx=ctx, silu=True)
        for attn in self.mid_block.attentions:
            attn.norm = MaskedGroupNorm(self.conv_norm_out_channels, ctx=ctx, silu=False, gathered=True)
        self.conv_norm_out = MaskedGroupNorm(self.conv_norm_out_channels, ctx=ctx, silu=True)
        self._constants: dict[tuple, ttnn.Tensor] = {}

    @property
    def conv_norm_out_channels(self) -> int:
        return self.conv_out.inner.in_channels

    def deallocate_weights(self) -> None:
        super().deallocate_weights()
        for t in self._constants.values():
            ttnn.deallocate(t)
        self._constants.clear()

    def _constant(self, shape: tuple[int, ...], value: float) -> ttnn.Tensor:
        key = (shape, value)
        if key not in self._constants:
            self._constants[key] = tensor.from_torch(
                torch.full(shape, value), device=self._ctx.device, mesh_axes=[None] * len(shape)
            )
        return self._constants[key]

    def _resnet(self, r: VaeResnetBlock, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        n1 = r.norm1.forward(x, masks.shard(int(x.shape[1]), int(x.shape[2])))
        c1 = r.conv1.forward(n1)
        ttnn.deallocate(n1)
        n2 = r.norm2.forward(c1, masks.shard(int(c1.shape[1]), int(c1.shape[2])))
        ttnn.deallocate(c1)
        h = r.conv2.forward(n2)
        ttnn.deallocate(n2)
        # the shortcut's output is the CCL manager's persistent reduce-scatter buffer: never freed here
        shortcut = r.conv_shortcut.forward(x) if r.conv_shortcut is not None else x
        out = shortcut + h
        ttnn.deallocate(h)
        return out

    def _attention(self, attn: VaeAttention, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        ctx, ccl, tp_axis = self._ctx, attn._ccl_manager, attn._tp_axis
        identity = x
        full = _all_gather_hw(ctx, x)
        n, h, w, _ = (int(d) for d in full.shape)
        xn = attn.norm.forward(full, masks.full(h, w))
        if full is not identity:
            ttnn.deallocate(full)
        xn = ttnn.reshape(xn, (n, 1, h * w, -1))
        if ccl is not None and tp_axis is not None:
            local_xn, xn = xn, ccl.all_gather(xn, dim=-1, mesh_axis=tp_axis, use_hyperparams=True)
            ttnn.deallocate(local_xn)
        q, k, v = attn.to_qkv.forward(xn, compute_kernel_config=attn._mm_compute_kernel_config)
        ttnn.deallocate(xn)
        if ccl is not None and tp_axis is not None:
            local_qkv = (q, k, v)
            q, k, v = (ccl.all_gather(t, dim=-1, mesh_axis=tp_axis, use_hyperparams=True) for t in local_qkv)
            for t in local_qkv:
                ttnn.deallocate(t)
        channels = int(q.shape[-1])
        program_config = attn._sdpa_program_config
        h_local = h // ctx.h_factor
        local = attn._tp_factor > 1 and h_local % attn._tp_factor == 0 and h_local // attn._tp_factor * w % 32 == 0
        if local:
            steps = [lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)]
            if ctx.h_factor > 1:
                steps.append(
                    lambda t: ttnn.mesh_partition(
                        t, dim=2, cluster_axis=ctx.h_mesh_axis, memory_config=t.memory_config()
                    )
                )
            steps.append(lambda t: ttnn.mesh_partition(t, dim=2, cluster_axis=tp_axis, memory_config=t.memory_config()))
            steps.append(lambda t: ttnn.to_layout(t, ttnn.TILE_LAYOUT))
            for step in steps:
                previous, q = q, step(q)
                ttnn.deallocate(previous)
        rows = int(q.shape[2])
        program_config = sdpa_config_for_rows(program_config, rows)
        # one extra tile column: the queries' 1 / 32 per column picks up the keys' bias, the values' zeros add nothing
        unextended = (q, k, v)
        q = ttnn.concat([q, self._constant((1, 1, rows, 32), 1.0 / 32)], dim=-1)
        k = ttnn.concat([k, masks.key_bias], dim=-1)
        v = ttnn.concat([v, self._constant((1, 1, h * w, 32), 0.0)], dim=-1)
        for t in unextended:
            ttnn.deallocate(t)
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=channels**-0.5,
            program_config=program_config,
            compute_kernel_config=attn._sdpa_compute_kernel_config,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o, extended = ttnn.slice(o, [0, 0, 0, 0], [n, 1, rows, channels]), o
        ttnn.deallocate(extended)
        if local:
            o = ccl.all_gather(o, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
            o = attn.to_out.forward(o, compute_kernel_config=attn._mm_compute_kernel_config)
            return ttnn.reshape(o, (n, h_local, w, -1)) + identity
        o = attn.to_out.forward(o, compute_kernel_config=attn._mm_compute_kernel_config)
        o = _partition_hw(ctx, ttnn.reshape(o, (n, h, w, -1)))
        return o + identity

    def forward_canvas(self, x: ttnn.Tensor, masks: CanvasMasks) -> ttnn.Tensor:
        x = self.conv_in.forward(x)
        for block in self.down_blocks:
            for r in block.resnets:
                x = self._resnet(r, x, masks)
            if block.downsampler is not None:
                if x.layout != ttnn.TILE_LAYOUT:
                    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                # zero right of and below the image: the strided conv's padding
                x = masks.shard(int(x.shape[1]), int(x.shape[2])).apply(x)
                x = block.downsampler.forward(x)
        mid = self.mid_block
        x = self._resnet(mid.resnets[0], x, masks)
        x = self._attention(mid.attentions[0], x, masks)
        x = self._resnet(mid.resnets[1], x, masks)
        x = self.conv_norm_out.forward(x, masks.shard(int(x.shape[1]), int(x.shape[2])))
        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            x = self._ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)
        x = self.conv_out.forward(x)
        return self.quant_conv.forward(x)


__all__ = ["CanvasMasks", "CanvasVaeEncoder", "MaskedGroupNorm", "canvas_for", "reference_canvases"]
