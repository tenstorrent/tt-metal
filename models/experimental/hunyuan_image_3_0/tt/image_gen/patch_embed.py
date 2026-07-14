# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 image patch-projection modules:
#   HunyuanTtUNetDown  (patch_embed)  -- VAE latent  -> transformer token seq
#   HunyuanTtUNetUp    (final_layer)  -- transformer tokens -> VAE latent / pred
#
# Mirrors ref/image_gen/patch_embed.py (UNetDown / UNetUp / ResBlock), which is
# a bit-exact extraction of upstream HunyuanImage3 (see that file for line refs).
#
# At the released config (patch_size=1) there is NO up/down-sampling: every
# ResBlock runs the simple updown=False path and the token grid equals the
# latent H x W. This port implements only that path.
#
# conv2d uses per-layer Conv2dConfig (sweep winners). GroupNorm uses the
# interleaved TILE manual path (fused HEIGHT GN removed — slower on grid-8).
# Residual add keeps the out_conv shard layout by aligning skip onto it.
#
# Timestep conditioning is adaptive group-norm: the (already-embedded) timestep
# vector -> SiLU -> Linear(emb, 2*out) -> (scale, shift); after the second
# GroupNorm,  h = norm(h) * (1 + scale) + shift.

import os

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from ..matmul_utils import act_width_sharded_linear
from .patch_embed_conv_configs import make_layer_conv2d_config

# Prefer sharded conv configs when HY_PATCH_EMBED_SHARDED=1 (set 0 for auto-only).
_PATCH_EMBED_SHARDED = os.environ.get("HY_PATCH_EMBED_SHARDED", "1") != "0"

# Emb-layer linear keeps HiFi4; convs use HiFi2 + bf16 dest for throughput.
_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

_CONV_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _conv2d_config(dtype, *, layer_name: str, sharded: bool) -> ttnn.Conv2dConfig:
    return make_layer_conv2d_config(layer_name, dtype=dtype, sharded=sharded)


def _to_row_major(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout == ttnn.ROW_MAJOR_LAYOUT:
        return x
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


def _to_interleaved_tile(x: ttnn.Tensor, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Single boundary convert: sharded/interleaved RM -> interleaved TILE (backbone scatter)."""
    if ttnn.is_sharded(x):
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if x.dtype != dtype:
        x = ttnn.typecast(x, dtype)
    return x


def _to_interleaved_dram(x: ttnn.Tensor) -> ttnn.Tensor:
    if ttnn.is_sharded(x):
        return ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x


def _match_shard_to(x: ttnn.Tensor, ref: ttnn.Tensor) -> ttnn.Tensor:
    """Reshard ``x`` onto ``ref``'s memory config (same layout → no-op)."""
    if not ttnn.is_sharded(ref):
        return _to_interleaved_dram(x)
    ref_mc = ref.memory_config()
    x = _to_row_major(x) if ref.layout == ttnn.ROW_MAJOR_LAYOUT else x
    if ref.layout == ttnn.TILE_LAYOUT and x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if ttnn.is_sharded(x) and x.memory_config().memory_layout == ref_mc.memory_layout:
        try:
            if (
                x.memory_config().shard_spec is not None
                and ref_mc.shard_spec is not None
                and x.memory_config().shard_spec == ref_mc.shard_spec
            ):
                return x
        except Exception:
            pass
    x = _to_interleaved_dram(x)
    return ttnn.interleaved_to_sharded(x, ref_mc)


# ---------------------------------------------------------------------------
# GroupNorm(32, C) on flat NHWC [1,1,B*H*W,C] — interleaved TILE manual path only.
# (Fused HEIGHT_SHARDED ttnn.group_norm removed: slower than manual on grid-8.)
# ---------------------------------------------------------------------------
class _TtGroupNorm:
    def __init__(self, device, num_channels, weight, bias, *, num_groups=32, eps=1e-5, dtype=ttnn.bfloat16):
        assert num_channels % 32 == 0 and num_channels % num_groups == 0
        self.device = device
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.cg = num_channels // num_groups
        self.eps = eps
        self.dtype = dtype
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, num_channels), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, num_channels), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x_flat, n_rows):
        C, G, Cg = self.num_channels, self.num_groups, self.cg
        x = _to_interleaved_dram(x_flat)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        xg = ttnn.reshape(x, [1, n_rows, G, Cg])

        m = ttnn.mean(xg, dim=3, keepdim=True)
        mean = ttnn.mean(m, dim=1, keepdim=True)
        ttnn.deallocate(m)
        sq = ttnn.multiply(xg, xg)
        s = ttnn.mean(sq, dim=3, keepdim=True)
        ttnn.deallocate(sq)
        msq = ttnn.mean(s, dim=1, keepdim=True)
        ttnn.deallocate(s)
        var = ttnn.subtract(msq, ttnn.multiply(mean, mean))
        ttnn.deallocate(msq)
        inv = ttnn.rsqrt(ttnn.add(var, self.eps))
        ttnn.deallocate(var)

        xn = ttnn.multiply(ttnn.subtract(xg, mean), inv)
        ttnn.deallocate(xg)
        ttnn.deallocate(mean)
        ttnn.deallocate(inv)

        xn = ttnn.reshape(xn, [1, 1, n_rows, C])
        out = ttnn.add(ttnn.multiply(xn, self.weight), self.bias)
        ttnn.deallocate(xn)
        return out

    def deallocate(self):
        ttnn.deallocate(self.weight)
        ttnn.deallocate(self.bias)


# ---------------------------------------------------------------------------
# Conv2d on flat NHWC [1,1,B*H*W,Cin] -> [1,1,B*Hout*Wout,Cout]
# ---------------------------------------------------------------------------
class _TtConv2d:
    def __init__(
        self,
        device,
        weight,
        bias,
        *,
        layer_name: str,
        stride=1,
        padding=1,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.layer_name = layer_name
        self.out_channels, self.in_channels, kh, kw = weight.shape
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.weight = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = _conv2d_config(dtype, layer_name=layer_name, sharded=_PATCH_EMBED_SHARDED)
        self._weights_prepared = False

    def __call__(self, x_flat, B, H, W):
        x = _to_interleaved_dram(_to_row_major(x_flat))
        if not self._weights_prepared:
            out, dim, wb = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.weight,
                bias_tensor=self.bias,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                batch_size=B,
                input_height=H,
                input_width=W,
                conv_config=self.conv_config,
                compute_config=_CONV_COMPUTE_KERNEL_CONFIG,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            Hout, Wout = dim
            w, b = wb
            if w is not self.weight:
                ttnn.deallocate(self.weight)
                self.weight = w
            if b is not None:
                if self.bias is not None and b is not self.bias:
                    ttnn.deallocate(self.bias)
                self.bias = b
            self._weights_prepared = True
            return out, Hout, Wout
        out, [Hout, Wout] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=_CONV_COMPUTE_KERNEL_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=False,
        )
        return out, Hout, Wout

    def deallocate(self):
        ttnn.deallocate(self.weight)
        if self.bias is not None:
            ttnn.deallocate(self.bias)


# ---------------------------------------------------------------------------
# ResBlock (updown=False)
# ---------------------------------------------------------------------------
class HunyuanTtResBlock(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        prefix,
        *,
        in_channels,
        out_channels,
        emb_channels,
        eps=1e-5,
        dtype=ttnn.bfloat16,
        conv_name_prefix: str = "",
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype

        g = lambda k: state_dict[f"{prefix}.{k}"]
        np = conv_name_prefix  # "" for patch_embed, "final_" for UNetUp resblock

        self.in_norm = _TtGroupNorm(
            device, in_channels, g("in_layers.0.weight"), g("in_layers.0.bias"), eps=eps, dtype=dtype
        )
        self.in_conv = _TtConv2d(
            device, g("in_layers.2.weight"), g("in_layers.2.bias"), layer_name=f"{np}in_conv", dtype=dtype
        )

        w = g("emb_layers.1.weight")
        b = g("emb_layers.1.bias")
        self.emb_w = ttnn.from_torch(
            w.transpose(0, 1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.emb_b = ttnn.from_torch(
            b.reshape(1, 1, 1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.out_norm = _TtGroupNorm(
            device, out_channels, g("out_layers.0.weight"), g("out_layers.0.bias"), eps=eps, dtype=dtype
        )
        self.out_conv = _TtConv2d(
            device, g("out_layers.3.weight"), g("out_layers.3.bias"), layer_name=f"{np}out_conv", dtype=dtype
        )

        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = _TtConv2d(
                device,
                g("skip_connection.weight"),
                g("skip_connection.bias"),
                layer_name=f"{np}skip_conv",
                stride=1,
                padding=0,
                dtype=dtype,
            )

    def __call__(self, x_flat, t_emb, B, H, W):
        n_rows = B * H * W
        sharded_path = _PATCH_EMBED_SHARDED

        h = self.in_norm(x_flat, n_rows)
        h = ttnn.silu(h)
        h, H, W = self.in_conv(h, B, H, W)
        n_rows = B * H * W

        e = ttnn.silu(t_emb)
        batch_rows = int(list(t_emb.shape)[-2])
        e = act_width_sharded_linear(
            e,
            self.emb_w,
            bias=self.emb_b,
            batch_rows=batch_rows,
            compute_kernel_config=_COMPUTE_KERNEL_CONFIG,
            device=self.device,
        )
        scale = ttnn.slice(e, [0, 0, 0, 0], [1, 1, B, self.out_channels])
        shift = ttnn.slice(e, [0, 0, 0, self.out_channels], [1, 1, B, 2 * self.out_channels])
        ttnn.deallocate(e)

        hn = self.out_norm(h, n_rows)
        ttnn.deallocate(h)
        scale = ttnn.reshape(scale, [1, 1, B, self.out_channels])
        shift = ttnn.reshape(shift, [1, 1, B, self.out_channels])
        scale_p1 = ttnn.add(scale, 1.0)
        ttnn.deallocate(scale)
        hn = ttnn.multiply(hn, scale_p1)
        ttnn.deallocate(scale_p1)
        hn = ttnn.add(hn, shift)
        ttnn.deallocate(shift)

        hn = ttnn.silu(hn)
        hn, H, W = self.out_conv(hn, B, H, W)

        if self.skip_conv is not None:
            skip, _, _ = self.skip_conv(x_flat, B, H, W)
        else:
            skip = x_flat
        if sharded_path and ttnn.is_sharded(hn):
            # Align skip onto out_conv's shard layout so residual add stays sharded.
            skip = _match_shard_to(skip, hn)
            out = ttnn.add(hn, skip)
        else:
            skip = ttnn.to_layout(_to_interleaved_dram(skip), ttnn.TILE_LAYOUT)
            hn = ttnn.to_layout(_to_interleaved_dram(hn), ttnn.TILE_LAYOUT)
            out = ttnn.add(hn, skip)
        ttnn.deallocate(hn)
        if self.skip_conv is not None or skip is not x_flat:
            ttnn.deallocate(skip)
        return out, H, W


# ---------------------------------------------------------------------------
# UNetDown (patch_embed)
# ---------------------------------------------------------------------------
class HunyuanTtUNetDown(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        prefix="patch_embed",
        *,
        patch_size=1,
        in_channels=32,
        emb_channels=4096,
        hidden_channels=1024,
        out_channels=4096,
        eps=1e-5,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        assert patch_size == 1, "only patch_size==1 is ported"
        self.device = device
        self.in_channels = in_channels

        self.conv0 = _TtConv2d(
            device,
            state_dict[f"{prefix}.model.0.weight"],
            state_dict[f"{prefix}.model.0.bias"],
            layer_name="conv0",
            dtype=dtype,
        )
        self.resblock = HunyuanTtResBlock(
            device,
            state_dict,
            f"{prefix}.model.1",
            in_channels=hidden_channels,
            out_channels=out_channels,
            emb_channels=emb_channels,
            eps=eps,
            dtype=dtype,
            conv_name_prefix="",
        )

    def __call__(self, x_bchw, t_emb):
        x_flat, B, H, W = _to_flat_nhwc(self.device, x_bchw, self.in_channels)
        return self.forward_latent(x_flat, t_emb, B, H, W)

    def forward_latent(self, x_flat, t_emb, B: int, H: int, W: int):
        x_flat, H, W = self.conv0(x_flat, B, H, W)
        out, H, W = self.resblock(x_flat, t_emb, B, H, W)
        ttnn.deallocate(x_flat)
        if _PATCH_EMBED_SHARDED:
            out = _to_interleaved_tile(out)
        return out, H, W


# ---------------------------------------------------------------------------
# UNetUp (final_layer)
# ---------------------------------------------------------------------------
class HunyuanTtUNetUp(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        prefix="final_layer",
        *,
        patch_size=1,
        in_channels=4096,
        emb_channels=4096,
        hidden_channels=1024,
        out_channels=32,
        out_norm=True,
        eps=1e-5,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        assert patch_size == 1, "only patch_size==1 is ported"
        assert out_norm, "final_layer uses out_norm=True"
        self.device = device
        self.in_channels = in_channels

        self.resblock = HunyuanTtResBlock(
            device,
            state_dict,
            f"{prefix}.model.0",
            in_channels=in_channels,
            out_channels=hidden_channels,
            emb_channels=emb_channels,
            eps=eps,
            dtype=dtype,
            conv_name_prefix="final_",
        )
        self.tail_norm = _TtGroupNorm(
            device,
            hidden_channels,
            state_dict[f"{prefix}.model.1.0.weight"],
            state_dict[f"{prefix}.model.1.0.bias"],
            eps=eps,
            dtype=dtype,
        )
        self.tail_conv = _TtConv2d(
            device,
            state_dict[f"{prefix}.model.1.2.weight"],
            state_dict[f"{prefix}.model.1.2.bias"],
            layer_name="final_tail_conv",
            dtype=dtype,
        )

    def __call__(self, tokens, t_emb, token_h, token_w, B=1):
        H, W = token_h, token_w
        h, H, W = self.resblock(tokens, t_emb, B, H, W)
        n_rows = B * H * W
        h = self.tail_norm(h, n_rows)
        h = ttnn.silu(h)
        out, H, W = self.tail_conv(h, B, H, W)
        return out, H, W


def _to_flat_nhwc(device, x, in_channels):
    """Accept torch [B,C,H,W], ttnn BTHWC, or flat NHWC; return (flat, B,H,W)."""
    if isinstance(x, torch.Tensor):
        B, C, H, W = x.shape
        assert C == in_channels
        nhwc = x.permute(0, 2, 3, 1).contiguous().reshape(1, 1, B * H * W, C)
        xf = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        return xf, B, H, W
    if isinstance(x, ttnn.Tensor):
        if len(x.shape) == 5:
            B, T, H, W, C = (int(x.shape[i]) for i in range(5))
            assert C == in_channels
            if T != 1:
                x = ttnn.slice(x, [0, 0, 0, 0, 0], [B, 1, H, W, C])
            xf = ttnn.reshape(x, [1, 1, B * H * W, C])
            return xf, B, H, W
        if len(x.shape) == 4 and int(x.shape[0]) == 1:
            B = 1
            _, n, C = (int(x.shape[i]) for i in range(3))
            H = W = int(n**0.5)
            return x, B, H, W
    raise TypeError("UNetDown entry: pass torch NCHW or ttnn BTHWC / flat NHWC")
