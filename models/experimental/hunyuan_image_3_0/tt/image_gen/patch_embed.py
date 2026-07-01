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
# Layout: everything runs in channels-last NHWC, kept flat as [1, 1, B*H*W, C]
# (ROW_MAJOR for conv2d, TILE for elementwise/groupnorm). The leading rearrange
# `b c h w -> b (h w) c` of UNetDown is then a no-op reshape; UNetUp's inverse
# `b (h w) c -> b c h w` is likewise a reshape because we stay channels-last.
#
# Timestep conditioning is adaptive group-norm: the (already-embedded) timestep
# vector -> SiLU -> Linear(emb, 2*out) -> (scale, shift); after the second
# GroupNorm,  h = norm(h) * (1 + scale) + shift.
#
# Weights (PyTorch checkpoint, NCHW / [out,in] linear):
#   patch_embed.model.0.{weight,bias}                  conv0  [hid,lat,3,3]
#   patch_embed.model.1.in_layers.0.{weight,bias}      GN(32,hid)
#   patch_embed.model.1.in_layers.2.{weight,bias}      conv  [out,hid,3,3]
#   patch_embed.model.1.emb_layers.1.{weight,bias}     Linear[2*out,emb]
#   patch_embed.model.1.out_layers.0.{weight,bias}     GN(32,out)
#   patch_embed.model.1.out_layers.3.{weight,bias}     conv  [out,out,3,3]
#   patch_embed.model.1.skip_connection.{weight,bias}  conv1x1 [out,in] (if in!=out)
# UNetUp (final_layer) mirrors this with an extra out-norm conv block at the end.

import ttnn
from models.common.lightweightmodule import LightweightModule


_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# ---------------------------------------------------------------------------
# GroupNorm(32, C) on a flat NHWC tensor [1,1,B*H*W,C].
#
# Computed manually (group mean/var reductions) rather than via the sharded
# ttnn.group_norm: the sharded path imposes a core-grid divisibility constraint
# that depends on the spatial size (num_virtual_rows <= Ht), which is brittle
# across the small test grid and the much larger real latent grids. The manual
# path is layout-agnostic and exact. Channels are the last (contiguous) dim, so
# group g owns the contiguous channel block [g*Cg : (g+1)*Cg].
# ---------------------------------------------------------------------------
class _TtGroupNorm:
    def __init__(self, device, num_channels, weight, bias, *, num_groups=32, eps=1e-5, dtype=ttnn.bfloat16):
        assert num_channels % 32 == 0 and num_channels % num_groups == 0
        self.device = device
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.cg = num_channels // num_groups
        self.eps = eps
        # affine weight/bias as per-channel row vectors [1,1,1,C]
        self.weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, num_channels), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, num_channels), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x_flat, n_rows):
        C, G, Cg = self.num_channels, self.num_groups, self.cg
        x = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(x, ttnn.float32)
        xg = ttnn.reshape(x, [1, n_rows, G, Cg])  # group channels along last dim

        # per-group mean over (spatial rows, in-group channels): reduce dim3 then dim1
        m = ttnn.mean(xg, dim=3, keepdim=True)  # [1,n,G,1]
        mean = ttnn.mean(m, dim=1, keepdim=True)  # [1,1,G,1]
        ttnn.deallocate(m)
        sq = ttnn.multiply(xg, xg)
        s = ttnn.mean(sq, dim=3, keepdim=True)
        ttnn.deallocate(sq)
        msq = ttnn.mean(s, dim=1, keepdim=True)  # [1,1,G,1]
        ttnn.deallocate(s)
        var = ttnn.subtract(msq, ttnn.multiply(mean, mean))  # [1,1,G,1]
        ttnn.deallocate(msq)
        inv = ttnn.rsqrt(ttnn.add(var, self.eps))  # [1,1,G,1]
        ttnn.deallocate(var)

        xn = ttnn.multiply(ttnn.subtract(xg, mean), inv)  # broadcast [1,1,G,1] over [1,n,G,Cg]
        ttnn.deallocate(xg)
        ttnn.deallocate(mean)
        ttnn.deallocate(inv)

        xn = ttnn.reshape(xn, [1, 1, n_rows, C])
        out = ttnn.add(ttnn.multiply(xn, self.weight), self.bias)  # per-channel affine
        ttnn.deallocate(xn)
        return out

    def deallocate(self):
        ttnn.deallocate(self.weight)
        ttnn.deallocate(self.bias)


# ---------------------------------------------------------------------------
# Conv2d on a flat NHWC tensor [1,1,B*H*W,Cin] -> [1,1,B*Hout*Wout,Cout]
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
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=dtype,
            shard_layout=None,
        )

    def __call__(self, x_flat, B, H, W):
        x = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT)
        out, [Hout, Wout], [self.weight, self.bias] = ttnn.conv2d(
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
        return out, Hout, Wout

    def deallocate(self):
        ttnn.deallocate(self.weight)
        if self.bias is not None:
            ttnn.deallocate(self.bias)


# ---------------------------------------------------------------------------
# ResBlock (updown=False) — the only path used at patch_size==1
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

        # in_layers: GroupNorm(32,in) -> SiLU -> Conv(in->out, 3x3)
        self.in_norm = _TtGroupNorm(
            device, in_channels, g("in_layers.0.weight"), g("in_layers.0.bias"), eps=eps, dtype=dtype
        )
        self.in_conv = _TtConv2d(device, g("in_layers.2.weight"), g("in_layers.2.bias"), dtype=dtype)

        # emb_layers: SiLU -> Linear(emb -> 2*out)
        w = g("emb_layers.1.weight")  # [2*out, emb]
        b = g("emb_layers.1.bias")  # [2*out]
        self.emb_w = ttnn.from_torch(
            w.transpose(0, 1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.emb_b = ttnn.from_torch(b.reshape(1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        # out_layers: GroupNorm(32,out) -> SiLU -> Conv(out->out, 3x3)
        self.out_norm = _TtGroupNorm(
            device, out_channels, g("out_layers.0.weight"), g("out_layers.0.bias"), eps=eps, dtype=dtype
        )
        self.out_conv = _TtConv2d(device, g("out_layers.3.weight"), g("out_layers.3.bias"), dtype=dtype)

        # skip: Identity if in==out else Conv1x1(in->out)
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = _TtConv2d(
                device, g("skip_connection.weight"), g("skip_connection.bias"), stride=1, padding=0, dtype=dtype
            )

    def __call__(self, x_flat, t_emb, B, H, W):
        n_rows = B * H * W

        # --- in_layers: norm -> silu -> conv ---
        h = self.in_norm(x_flat, n_rows)
        h = ttnn.silu(h)
        h, H, W = self.in_conv(h, B, H, W)  # [1,1,n,out]
        n_rows = B * H * W

        # --- timestep -> (scale, shift) ---
        e = ttnn.silu(t_emb)  # [1,1,N,emb] (N==B)
        e = ttnn.linear(e, self.emb_w, bias=self.emb_b, compute_kernel_config=_COMPUTE_KERNEL_CONFIG)  # [1,1,B,2*out]
        scale = ttnn.slice(e, [0, 0, 0, 0], [1, 1, B, self.out_channels])
        shift = ttnn.slice(e, [0, 0, 0, self.out_channels], [1, 1, B, 2 * self.out_channels])
        ttnn.deallocate(e)

        # --- out_norm then adaGN modulation: norm(h)*(1+scale)+shift ---
        hn = self.out_norm(h, n_rows)
        ttnn.deallocate(h)
        # scale/shift broadcast over spatial: reshape to [1,1,1,out] (B==1) for bcast
        scale = ttnn.reshape(scale, [1, 1, B, self.out_channels])
        shift = ttnn.reshape(shift, [1, 1, B, self.out_channels])
        scale_p1 = ttnn.add(scale, 1.0)
        ttnn.deallocate(scale)
        hn = ttnn.multiply(hn, scale_p1)
        ttnn.deallocate(scale_p1)
        hn = ttnn.add(hn, shift)
        ttnn.deallocate(shift)

        # --- out_rest: silu -> conv ---
        hn = ttnn.silu(hn)
        hn, H, W = self.out_conv(hn, B, H, W)

        # --- skip ---
        if self.skip_conv is not None:
            skip, _, _ = self.skip_conv(x_flat, B, H, W)
        else:
            skip = x_flat
        skip = ttnn.to_layout(skip, ttnn.TILE_LAYOUT)
        hn = ttnn.to_layout(hn, ttnn.TILE_LAYOUT)
        out = ttnn.add(hn, skip)
        ttnn.deallocate(hn)
        if self.skip_conv is not None:
            ttnn.deallocate(skip)
        return out, H, W


# ---------------------------------------------------------------------------
# UNetDown (patch_embed): Conv(lat->hid) -> ResBlock(hid->out) -> tokens
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
        """x_bchw: torch [B, in_ch, H, W] or ttnn NHWC flat [1,1,B*H*W,in_ch].
        Returns (tokens [1,1,B*H*W,out], token_h, token_w)."""
        x_flat, B, H, W = _to_flat_nhwc(self.device, x_bchw, self.in_channels)
        return self.forward_latent(x_flat, t_emb, B, H, W)

    def forward_latent(self, x_flat, t_emb, B: int, H: int, W: int):
        """Run conv+resblock from flat NHWC ``[1,1,B*H*W,C]`` (already on device)."""
        x_flat, H, W = self.conv0(x_flat, B, H, W)
        out, H, W = self.resblock(x_flat, t_emb, B, H, W)
        ttnn.deallocate(x_flat)
        return out, H, W


# ---------------------------------------------------------------------------
# UNetUp (final_layer): tokens -> ResBlock(in->hid) -> out_norm conv -> latent
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
        # model.1 = Sequential(GroupNorm(32,hid), SiLU, Conv(hid->out,3x3))
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
        """tokens: ttnn [1,1,B*H*W,in] (== `b (h w) c`). Returns NHWC flat latent
        [1,1,B*H*W,out] plus (H,W)."""
        H, W = token_h, token_w
        h, H, W = self.resblock(tokens, t_emb, B, H, W)
        n_rows = B * H * W
        h = self.tail_norm(h, n_rows)
        h = ttnn.silu(h)
        out, H, W = self.tail_conv(h, B, H, W)
        return out, H, W


# ---------------------------------------------------------------------------
def _to_flat_nhwc(device, x, in_channels):
    """Accept torch [B,C,H,W], ttnn BTHWC ``[B,1,H,W,C]``, or flat NHWC; return (flat, B,H,W)."""
    import torch

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
