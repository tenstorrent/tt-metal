# SPDX-License-Identifier: Apache-2.0
"""ttnn port of MoGe-2's ConvStack decoder (neck + points/normal/mask heads), on device.

Faithfully mirrors moge.model.modules.ConvStack / ResidualConvBlock / Resampler:
  - input_blocks[i]:  Conv1x1 (pointwise -> ttnn.linear over channels) or Identity
  - res_blocks[i]:    n_i x ResidualConvBlock = x + conv3x3(relu(conv3x3(relu(x))))   (norm='none')
  - resamplers[i]:    conv_transpose: ConvT2d(k2,s2) -> Conv3x3 ;  bilinear: Upsample(bilinear,2) -> Conv3x3
  - output_blocks[i]: Conv1x1 or Identity
  - forward: x = in[0]; per level: x (+ input_blocks[i](in[i])) -> res_blocks[i] -> out[i]; if i<L-1: x = resamplers[i](x)

Tensors flow channels-last as (ttnn [1,1,H*W,C], H, W). 3x3 convs use zero padding
(ttnn.conv2d) vs the reference's 'replicate'; the border difference is negligible at
the global-PCC scale (huge accuracy headroom). Optimization-phase; PCC-gated.
"""
import torch
import ttnn


def _to_cl(t_nchw, device, dtype=ttnn.bfloat16):
    """torch [B,C,H,W] -> (ttnn [1,1,H*W,C] TILE, H, W)."""
    B, C, H, W = t_nchw.shape
    x = t_nchw.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C).contiguous()
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device), H, W


class _Conv:
    """Lazy ttnn.conv2d holder (weights prepared on first call & cached)."""
    def __init__(self, conv, device, compute_config, wdtype=ttnn.bfloat8_b):
        self.cin = conv.in_channels
        self.cout = conv.out_channels
        self.k = conv.kernel_size
        self.s = conv.stride
        self.p = conv.padding
        self.w = ttnn.from_torch(conv.weight.detach(), dtype=ttnn.bfloat16)  # [Cout,Cin,kh,kw] host
        self.b = ttnn.from_torch(conv.bias.detach().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16) if conv.bias is not None else None
        self.device = device
        self.cc = compute_config
        self.wdtype = wdtype

    def __call__(self, x, H, W):
        pmode = ttnn.PaddingMode.Replicate if (self.p[0] > 0 or self.p[1] > 0) else ttnn.PaddingMode.Zeros
        out, [pw, pb] = ttnn.conv2d(
            input_tensor=x, weight_tensor=self.w, bias_tensor=self.b, device=self.device,
            in_channels=self.cin, out_channels=self.cout, kernel_size=self.k, stride=self.s, padding=self.p,
            batch_size=1, input_height=H, input_width=W,
            conv_config=ttnn.Conv2dConfig(weights_dtype=self.wdtype, padding_mode=pmode),
            compute_config=self.cc, groups=1, return_weights_and_bias=True, return_output_dim=False,
        )
        self.w, self.b = pw, pb
        Ho = (H + 2 * self.p[0] - self.k[0]) // self.s[0] + 1
        Wo = (W + 2 * self.p[1] - self.k[1]) // self.s[1] + 1
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(ttnn.to_layout(out, ttnn.TILE_LAYOUT), (1, 1, Ho * Wo, self.cout))
        return out, Ho, Wo


class _ConvT:
    """ConvTranspose2d (kernel=stride=2) via ttnn.conv_transpose2d."""
    def __init__(self, conv, device, compute_config, wdtype=ttnn.bfloat8_b):
        self.cin, self.cout = conv.in_channels, conv.out_channels
        self.k, self.s = conv.kernel_size, conv.stride
        self.w = ttnn.from_torch(conv.weight.detach(), dtype=ttnn.bfloat16)  # [Cin,Cout,kh,kw]
        self.b = ttnn.from_torch(conv.bias.detach().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16) if conv.bias is not None else None
        self.device, self.cc, self.wdtype = device, compute_config, wdtype

    def __call__(self, x, H, W):
        out, [pw, pb] = ttnn.conv_transpose2d(
            input_tensor=x, weight_tensor=self.w, bias_tensor=self.b, device=self.device,
            in_channels=self.cin, out_channels=self.cout, kernel_size=self.k, stride=self.s,
            padding=(0, 0), output_padding=(0, 0), batch_size=1, input_height=H, input_width=W,
            conv_config=ttnn.Conv2dConfig(weights_dtype=self.wdtype),
            compute_config=self.cc, groups=1, return_weights_and_bias=True, return_output_dim=False,
        )
        self.w, self.b = pw, pb  # cache prepared weights (no host write on later/traced calls)
        Ho, Wo = H * self.s[0], W * self.s[1]
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(ttnn.to_layout(out, ttnn.TILE_LAYOUT), (1, 1, Ho * Wo, self.cout))
        return out, Ho, Wo


class _Pointwise:
    """Conv2d 1x1 as ttnn.linear over the channel dim."""
    def __init__(self, conv, device, wdtype=ttnn.bfloat8_b):
        w = conv.weight.detach().squeeze(-1).squeeze(-1)  # [Cout,Cin]
        self.w = ttnn.from_torch(w.t().contiguous(), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.b = ttnn.from_torch(conv.bias.detach().reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if conv.bias is not None else None
        self.cc = None

    def __call__(self, x, H, W):
        return ttnn.linear(x, self.w, bias=self.b), H, W


class _ResBlock:
    """ResidualConvBlock with norm='none': x + conv3x3(relu(conv3x3(relu(x))))."""
    def __init__(self, block, device, compute_config):
        self.conv1 = _Conv(block.layers[2], device, compute_config)
        self.conv2 = _Conv(block.layers[5], device, compute_config)
        # skip is Identity (in==out in neck/heads res blocks); guard anyway
        self.skip = None
        if not isinstance(block.skip_connection, torch.nn.Identity):
            self.skip = _Pointwise(block.skip_connection, device)

    def __call__(self, x, H, W):
        skip = x if self.skip is None else self.skip(x, H, W)[0]
        h = ttnn.relu(x)
        h, H, W = self.conv1(h, H, W)
        h = ttnn.relu(h)
        h, H, W = self.conv2(h, H, W)
        return ttnn.add(h, skip), H, W


class _Resampler:
    def __init__(self, resampler, device, compute_config):
        mods = list(resampler)
        if isinstance(mods[0], torch.nn.Upsample):           # bilinear: Upsample -> Conv3x3
            self.kind = "bilinear"
            self.scale = int(mods[0].scale_factor)
            self.conv = _Conv(mods[1], device, compute_config)
        else:                                                # conv_transpose: ConvT -> Conv3x3
            self.kind = "conv_transpose"
            self.convt = _ConvT(mods[0], device, compute_config)
            self.conv = _Conv(mods[1], device, compute_config)

    def __call__(self, x, H, W):
        if self.kind == "conv_transpose":
            x, H, W = self.convt(x, H, W)
        else:
            C = x.shape[-1]
            xs = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, H, W, C))
            xs = ttnn.upsample(xs, self.scale, mode="bilinear")
            H, W = H * self.scale, W * self.scale
            if xs.is_sharded():
                xs = ttnn.sharded_to_interleaved(xs, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(xs, (1, 1, H * W, C))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return self.conv(x, H, W)


class TtConvStack:
    def __init__(self, convstack, device, compute_config):
        self.device = device
        self.n = len(convstack.res_blocks)
        self.input_blocks = [None if isinstance(m, torch.nn.Identity) else _Pointwise(m, device) for m in convstack.input_blocks]
        self.output_blocks = [None if isinstance(m, torch.nn.Identity) else _Pointwise(m, device) for m in convstack.output_blocks]
        self.res_blocks = [[_ResBlock(b, device, compute_config) for b in seq] for seq in convstack.res_blocks]
        self.resamplers = [_Resampler(r, device, compute_config) for r in convstack.resamplers]

    def __call__(self, in_features):
        """in_features: list of (ttnn [1,1,H*W,C], H, W) or None. -> list of (ttnn, H, W)."""
        out = []
        x = xH = xW = None
        for i in range(self.n):
            feat = in_features[i]
            if feat is not None:
                t, H, W = feat
                if self.input_blocks[i] is not None:
                    t, H, W = self.input_blocks[i](t, H, W)
                if i == 0:
                    x, xH, xW = t, H, W
                else:
                    x = ttnn.add(x, t)
            for rb in self.res_blocks[i]:
                x, xH, xW = rb(x, xH, xW)
            o, oH, oW = x, xH, xW
            if self.output_blocks[i] is not None:
                o, oH, oW = self.output_blocks[i](o, xH, xW)
            out.append((o, oH, oW))
            if i < self.n - 1:
                x, xH, xW = self.resamplers[i](x, xH, xW)
        return out
