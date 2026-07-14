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
# conv2d auto-shards when optimal. Fused sharded ttnn.group_norm runs when the
# activation is sharded, avoiding ShardedToInterleaved + manual TILE reduces.
#
# Timestep conditioning is adaptive group-norm: the (already-embedded) timestep
# vector -> SiLU -> Linear(emb, 2*out) -> (scale, shift); after the second
# GroupNorm,  h = norm(h) * (1 + scale) + shift.

import os

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from ..matmul_utils import act_width_sharded_linear

# HEIGHT_SHARDED conv + sharded group_norm (set HY_PATCH_EMBED_SHARDED=0 for legacy interleaved GN).
_PATCH_EMBED_SHARDED = os.environ.get("HY_PATCH_EMBED_SHARDED", "1") != "0"

_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _conv2d_config(dtype, *, sharded: bool) -> ttnn.Conv2dConfig:
    # Keep conv2d defaults (auto shard when optimal). The win is sharded GN consuming
    # sharded conv output without ShardedToInterleaved + manual TILE reduces.
    return ttnn.Conv2dConfig(weights_dtype=dtype, shard_layout=None)


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


# ---------------------------------------------------------------------------
# GroupNorm(32, C) on flat NHWC [1,1,B*H*W,C].
# Sharded path: fused ttnn.group_norm on HEIGHT_SHARDED ROW_MAJOR (conv output).
# Legacy path: manual stats on interleaved TILE (HY_PATCH_EMBED_SHARDED=0).
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
        self._weight_host = weight.detach().float().reshape(-1)
        self._bias_host = bias.detach().float().reshape(-1)
        self._sharded_cache: dict[int, tuple] = {}
        # Legacy interleaved TILE affine (manual path).
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, num_channels), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, num_channels), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def _sharded_state(self, n_rows: int):
        cached = self._sharded_cache.get(n_rows)
        if cached is not None:
            return cached
        mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
            device=self.device,
            num_channels=self.num_channels,
            num_groups=self.num_groups,
            input_nhw=n_rows,
            is_height_sharded=True,
            is_row_major=True,
        )
        num_cores = ttnn.get_group_norm_cores_across_channel(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            grid_size,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(
                num_channel=self.num_channels,
                num_groups=self.num_groups,
                num_cores_across_channel=num_cores,
                data_type=self.dtype,
            ),
            self.device,
        )
        w_rm = ttnn.create_group_norm_weight_bias_rm(
            input_tensor=self._weight_host.to(torch.bfloat16),
            num_channels=self.num_channels,
            num_cores_x=num_cores,
        )
        b_rm = ttnn.create_group_norm_weight_bias_rm(
            input_tensor=self._bias_host.to(torch.bfloat16),
            num_channels=self.num_channels,
            num_cores_x=num_cores,
        )
        gamma = ttnn.from_torch(
            w_rm,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta = ttnn.from_torch(
            b_rm,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cached = (mask, gamma, beta, mem_config, grid_size)
        self._sharded_cache[n_rows] = cached
        return cached

    def _forward_sharded(self, x_flat: ttnn.Tensor, n_rows: int) -> ttnn.Tensor:
        mask, gamma, beta, mem_config, grid_size = self._sharded_state(n_rows)
        return ttnn.group_norm(
            x_flat,
            num_groups=self.num_groups,
            input_mask=mask,
            weight=gamma,
            bias=beta,
            epsilon=self.eps,
            memory_config=mem_config,
            core_grid=grid_size,
        )

    def _forward_interleaved(self, x_flat: ttnn.Tensor, n_rows: int) -> ttnn.Tensor:
        C, G, Cg = self.num_channels, self.num_groups, self.cg
        x = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)
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

    def __call__(self, x_flat, n_rows):
        # Conv outputs are ROW_MAJOR; use fused sharded GN whenever sharded mode is on.
        # TILE inputs (e.g. UNetUp tokens from backbone) keep the interleaved manual path.
        if _PATCH_EMBED_SHARDED and x_flat.layout == ttnn.ROW_MAJOR_LAYOUT:
            return self._forward_sharded(x_flat, n_rows)
        return self._forward_interleaved(x_flat, n_rows)

    def deallocate(self):
        ttnn.deallocate(self.weight)
        ttnn.deallocate(self.bias)


# ---------------------------------------------------------------------------
# Conv2d on flat NHWC [1,1,B*H*W,Cin] -> [1,1,B*Hout*Wout,Cout]
# ---------------------------------------------------------------------------
class _TtConv2d:
    def __init__(self, device, weight, bias, *, stride=1, padding=1, dtype=ttnn.bfloat16):
        self.device = device
        self.out_channels, self.in_channels, kh, kw = weight.shape
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.weight = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = _conv2d_config(dtype, sharded=_PATCH_EMBED_SHARDED)
        self._weights_prepared = False

    def __call__(self, x_flat, B, H, W):
        x = _to_row_major(x_flat)
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
                compute_config=_COMPUTE_KERNEL_CONFIG,
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
            compute_config=_COMPUTE_KERNEL_CONFIG,
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
        self, device, state_dict, prefix, *, in_channels, out_channels, emb_channels, eps=1e-5, dtype=ttnn.bfloat16
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype

        g = lambda k: state_dict[f"{prefix}.{k}"]

        self.in_norm = _TtGroupNorm(
            device, in_channels, g("in_layers.0.weight"), g("in_layers.0.bias"), eps=eps, dtype=dtype
        )
        self.in_conv = _TtConv2d(device, g("in_layers.2.weight"), g("in_layers.2.bias"), dtype=dtype)

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
        self.out_conv = _TtConv2d(device, g("out_layers.3.weight"), g("out_layers.3.bias"), dtype=dtype)

        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = _TtConv2d(
                device, g("skip_connection.weight"), g("skip_connection.bias"), stride=1, padding=0, dtype=dtype
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
        if sharded_path and ttnn.is_sharded(hn) and ttnn.is_sharded(skip):
            out = ttnn.add(hn, skip)
        else:
            skip = ttnn.to_layout(skip, ttnn.TILE_LAYOUT)
            hn = ttnn.to_layout(hn, ttnn.TILE_LAYOUT)
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
            device, state_dict[f"{prefix}.model.0.weight"], state_dict[f"{prefix}.model.0.bias"], dtype=dtype
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
            device, state_dict[f"{prefix}.model.1.2.weight"], state_dict[f"{prefix}.model.1.2.bias"], dtype=dtype
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
