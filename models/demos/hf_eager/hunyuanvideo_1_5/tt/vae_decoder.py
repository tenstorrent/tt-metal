# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of diffusers `AutoencoderKLHunyuanVideo15` decoder (HunyuanVideo15Decoder3D).

Built bottom-up, each block PCC-validated against the CPU reference:
  CausalConv3d -> RMS_norm -> ResnetBlock -> AttnBlock -> Upsample(DCAE) -> Decoder.

Tensors are carried in ttnn BTHWC (B, T, H, W, C) ROW_MAJOR layout to match
`ttnn.experimental.conv3d`. Torch reference uses NCTHW (B, C, T, H, W).
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.utils.conv3d import get_conv3d_config, register_conv3d_configs

# Conservative conv3d blockings for the decoder's channel combos. The util's
# fallback uses C_in_block = in_channels, which overflows L1 for the wide
# channel-expand upsample convs (e.g. 1024->8192). Cap C_in_block small; these
# are correctness-first (untuned) and can be re-swept for speed later.
_K = (3, 3, 3)
register_conv3d_configs(
    {
        (32, 1024, _K): (32, 32, 1, 1, 1),
        (1024, 1024, _K): (128, 32, 1, 1, 1),
        (1024, 8192, _K): (128, 32, 1, 1, 1),
        (1024, 4096, _K): (128, 32, 1, 1, 1),
        (512, 512, _K): (128, 32, 1, 1, 1),
        (512, 1024, _K): (128, 32, 1, 1, 1),
        (512, 2048, _K): (128, 32, 1, 1, 1),
        (256, 256, _K): (128, 32, 1, 1, 1),
        (256, 512, _K): (128, 32, 1, 1, 1),
        (256, 1024, _K): (128, 32, 1, 1, 1),
        (128, 128, _K): (128, 32, 1, 1, 1),
        (128, 512, _K): (128, 32, 1, 1, 1),
        (128, 3, _K): (128, 32, 1, 1, 1),
    }
)


def replicate_pad_bthwc(x, t_front, hpad, wpad):
    """Replicate-pad a (B,T,H,W,C) ROW_MAJOR tensor: T front-only, H/W both sides.

    Matches diffusers HunyuanVideo15CausalConv3d's F.pad(mode="replicate", (W,W,H,H,T,0)).
    """
    B, T, H, W, C = x.shape
    if wpad > 0:
        left = ttnn.slice(x, [0, 0, 0, 0, 0], [B, T, H, 1, C])
        right = ttnn.slice(x, [0, 0, 0, W - 1, 0], [B, T, H, W, C])
        x = ttnn.concat([left] * wpad + [x] + [right] * wpad, dim=3)
        B, T, H, W, C = x.shape
    if hpad > 0:
        top = ttnn.slice(x, [0, 0, 0, 0, 0], [B, T, 1, W, C])
        bot = ttnn.slice(x, [0, 0, H - 1, 0, 0], [B, T, H, W, C])
        x = ttnn.concat([top] * hpad + [x] + [bot] * hpad, dim=2)
        B, T, H, W, C = x.shape
    if t_front > 0:
        f0 = ttnn.slice(x, [0, 0, 0, 0, 0], [B, 1, H, W, C])
        x = ttnn.concat([f0] * t_front + [x], dim=1)
    return x


class CausalConv3d:
    """ttnn HunyuanVideo15CausalConv3d: replicate-pad (on device) + conv3d(pad=0)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, *, device, dtype=ttnn.bfloat16):
        # weight: torch (Cout, Cin, kt, kh, kw)
        cout, cin, kt, kh, kw = weight.shape
        assert kt == kh == kw, f"only cubic kernels supported, got {(kt, kh, kw)}"
        self.k = kt
        self.cout = cout
        self.device = device
        self.dtype = dtype
        self.t_front = kt - 1
        self.pad_hw = kt // 2

        self.cfg = get_conv3d_config(cin, cout, (kt, kh, kw), dtype, device.compute_with_storage_grid_size())
        w = ttnn.from_torch(weight, dtype=dtype)
        w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=w, C_in_block=self.cfg.C_in_block, device=device)
        if not ttnn.is_tensor_storage_on_device(w):
            w = ttnn.to_device(w, device)
        self.w = w
        if bias is None:
            bias = torch.zeros(cout)
        self.b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x_bthwc):
        if self.t_front or self.pad_hw:
            x_bthwc = replicate_pad_bthwc(x_bthwc, self.t_front, self.pad_hw, self.pad_hw)
        return ttnn.experimental.conv3d(
            input_tensor=x_bthwc,
            weight_tensor=self.w,
            bias_tensor=self.b,
            device=self.device,
            config=self.cfg,
            output_channels=self.cout,
            kernel_size=(self.k, self.k, self.k),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.ckc,
        )


class RMSNorm:
    """ttnn HunyuanVideo15RMS_norm (channel_first, images=False, bias=False).

    Reference does F.normalize(x, dim=channel) * sqrt(dim) * gamma, i.e. RMS-norm
    across the channel dim (mean-square over C), scaled by per-channel gamma.
    In BTHWC the channel dim is last, so this is an RMS over the last axis.
    """

    def __init__(self, gamma: torch.Tensor, *, device, dtype=ttnn.bfloat16, eps=1e-12):
        g = gamma.reshape(-1).float()  # (C,)
        self.C = g.numel()
        self.eps = eps
        # broadcastable over (B,T,H,W,C): shape (1,1,1,1,C), TILE layout for elementwise on last dim
        self.gamma = ttnn.from_torch(g.reshape(1, 1, 1, 1, self.C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.dtype = dtype

    def __call__(self, x_bthwc):
        xt = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
        ms = ttnn.mean(ttnn.mul(xt, xt), dim=-1, keepdim=True)  # mean-square over channel
        inv = ttnn.rsqrt(ttnn.add(ms, self.eps))
        out = ttnn.mul(ttnn.mul(xt, inv), self.gamma)
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


def silu(x_bthwc):
    xt = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
    xt = ttnn.silu(xt)
    return ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)


def _block_causal_mask(n_frame, n_hw, dtype=torch.float32):
    """Additive (seq,seq) mask: token i (in frame f) attends to all tokens in frames 0..f."""
    seq = n_frame * n_hw
    mask = torch.full((seq, seq), float("-inf"), dtype=dtype)
    for i in range(seq):
        f = i // n_hw
        mask[i, : (f + 1) * n_hw] = 0.0
    return mask


class AttnBlock:
    """ttnn HunyuanVideo15AttnBlock: RMSNorm -> 1x1 q/k/v -> block-causal attention -> 1x1 proj + identity."""

    def __init__(self, torch_attn, *, device, dtype=ttnn.bfloat16):
        sd = torch_attn.state_dict()
        self.device = device
        self.dtype = dtype
        self.norm = RMSNorm(sd["norm.gamma"], device=device, dtype=dtype)
        self.to_q = CausalConv3d(sd["to_q.weight"], sd["to_q.bias"], device=device, dtype=dtype)  # k=1
        self.to_k = CausalConv3d(sd["to_k.weight"], sd["to_k.bias"], device=device, dtype=dtype)
        self.to_v = CausalConv3d(sd["to_v.weight"], sd["to_v.bias"], device=device, dtype=dtype)
        self.proj_out = CausalConv3d(sd["proj_out.weight"], sd["proj_out.bias"], device=device, dtype=dtype)
        self.C = int(torch_attn.in_channels)
        self.scale = self.C**-0.5

    def __call__(self, x_bthwc):
        identity = x_bthwc
        h = self.norm(x_bthwc)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        B, T, H, W, C = q.shape
        seq = T * H * W
        mask = ttnn.from_torch(
            _block_causal_mask(T, H * W), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        def flat(t):
            return ttnn.to_layout(ttnn.reshape(t, (B, seq, C)), ttnn.TILE_LAYOUT)

        q2, k2, v2 = flat(q), flat(k), flat(v)
        scores = ttnn.matmul(q2, ttnn.transpose(k2, -2, -1))  # (B, seq, seq)
        scores = ttnn.add(ttnn.mul(scores, self.scale), mask)
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v2)  # (B, seq, C)
        out = ttnn.to_layout(ttnn.reshape(out, (B, T, H, W, C)), ttnn.ROW_MAJOR_LAYOUT)
        out = self.proj_out(out)
        return ttnn.add(out, identity)


def _dcae_upsample_rearrange(x_bthwc, r1, r2, r3):
    """(b, f, h, w, r1*r2*r3*c) -> (b, r1*f, r2*h, r3*w, c). BTHWC analogue of the
    reference NCTHW view/permute (channel-packed r1->T, r2->H, r3->W)."""
    b, f, h, w, pc = x_bthwc.shape
    c = pc // (r1 * r2 * r3)
    x = ttnn.reshape(x_bthwc, (b, f, h, w, r1, r2, r3, c))
    x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))  # (b, f, r1, h, r2, w, r3, c)
    return ttnn.reshape(x, (b, f * r1, h * r2, w * r3, c))


class Upsample:
    """ttnn HunyuanVideo15Upsample (DCAE): channel-expand CausalConv3d + rearrange + residual.

    Temporal variant (add_temporal_upsample) doubles frames [1:] but leaves frame 0
    spatial-only, to preserve temporal causality (the reference's asymmetric first frame).
    """

    def __init__(self, torch_up, *, device, dtype=ttnn.bfloat16):
        w = torch_up.conv.conv.weight.detach()
        b = torch_up.conv.conv.bias.detach() if torch_up.conv.conv.bias is not None else None
        self.conv = CausalConv3d(w, b, device=device, dtype=dtype)
        self.add_temporal = bool(torch_up.add_temporal_upsample)
        self.repeats = int(torch_up.repeats)

    def __call__(self, x_bthwc):
        h = self.conv(x_bthwc)
        if self.add_temporal:
            B, T, H, W, PC = h.shape
            h_first = ttnn.slice(h, [0, 0, 0, 0, 0], [B, 1, H, W, PC])
            h_first = _dcae_upsample_rearrange(h_first, 1, 2, 2)
            c_r = h_first.shape[-1]
            h_first = ttnn.slice(
                h_first,
                [0, 0, 0, 0, 0],
                [h_first.shape[0], h_first.shape[1], h_first.shape[2], h_first.shape[3], c_r // 2],
            )
            h_next = ttnn.slice(h, [0, 1, 0, 0, 0], [B, T, H, W, PC])
            h_next = _dcae_upsample_rearrange(h_next, 2, 2, 2)
            h = ttnn.concat([h_first, h_next], dim=1)

            Bx, Tx, Hx, Wx, Cx = x_bthwc.shape
            x_first = ttnn.slice(x_bthwc, [0, 0, 0, 0, 0], [Bx, 1, Hx, Wx, Cx])
            x_first = _dcae_upsample_rearrange(x_first, 1, 2, 2)
            x_first = ttnn.repeat_interleave(x_first, self.repeats // 2, dim=4)
            x_next = ttnn.slice(x_bthwc, [0, 1, 0, 0, 0], [Bx, Tx, Hx, Wx, Cx])
            x_next = _dcae_upsample_rearrange(x_next, 2, 2, 2)
            x_next = ttnn.repeat_interleave(x_next, self.repeats, dim=4)
            shortcut = ttnn.concat([x_first, x_next], dim=1)
        else:
            h = _dcae_upsample_rearrange(h, 1, 2, 2)
            shortcut = ttnn.repeat_interleave(x_bthwc, self.repeats, dim=4)
            shortcut = _dcae_upsample_rearrange(shortcut, 1, 2, 2)
        return ttnn.add(h, shortcut)


class ResnetBlock:
    """ttnn HunyuanVideo15ResnetBlock: norm-silu-conv x2 + (1x1 conv shortcut if C changes)."""

    def __init__(self, torch_block, *, device, dtype=ttnn.bfloat16):
        sd = torch_block.state_dict()
        self.norm1 = RMSNorm(sd["norm1.gamma"], device=device, dtype=dtype)
        self.conv1 = CausalConv3d(sd["conv1.conv.weight"], sd["conv1.conv.bias"], device=device, dtype=dtype)
        self.norm2 = RMSNorm(sd["norm2.gamma"], device=device, dtype=dtype)
        self.conv2 = CausalConv3d(sd["conv2.conv.weight"], sd["conv2.conv.bias"], device=device, dtype=dtype)
        self.conv_shortcut = None
        if "conv_shortcut.weight" in sd:
            self.conv_shortcut = CausalConv3d(
                sd["conv_shortcut.weight"], sd.get("conv_shortcut.bias"), device=device, dtype=dtype
            )

    def __call__(self, x_bthwc):
        residual = x_bthwc
        h = self.norm1(x_bthwc)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return ttnn.add(h, residual)


class MidBlock:
    """ttnn HunyuanVideo15MidBlock: resnet -> (attn -> resnet)*."""

    def __init__(self, torch_mid, *, device, dtype=ttnn.bfloat16):
        self.resnets = [ResnetBlock(r, device=device, dtype=dtype) for r in torch_mid.resnets]
        self.attentions = [
            AttnBlock(a, device=device, dtype=dtype) if a is not None else None for a in torch_mid.attentions
        ]

    def __call__(self, h):
        h = self.resnets[0](h)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                h = attn(h)
            h = resnet(h)
        return h


class UpBlock3D:
    """ttnn HunyuanVideo15UpBlock3D: resnets then optional upsampler."""

    def __init__(self, torch_up, *, device, dtype=ttnn.bfloat16):
        self.resnets = [ResnetBlock(r, device=device, dtype=dtype) for r in torch_up.resnets]
        self.upsamplers = (
            [Upsample(u, device=device, dtype=dtype) for u in torch_up.upsamplers]
            if torch_up.upsamplers is not None
            else []
        )

    def __call__(self, h):
        for r in self.resnets:
            h = r(h)
        for u in self.upsamplers:
            h = u(h)
        return h


class HunyuanVideo15Decoder:
    """ttnn port of HunyuanVideo15Decoder3D. Input/output in BTHWC.

    Build from the torch decoder module:  dec_tt = HunyuanVideo15Decoder(vae.decoder, device=mesh)
    Call with a BTHWC latent (B, T, H, W, 32); returns BTHWC video (B, T', H', W', 3).
    """

    def __init__(self, torch_dec, *, device, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype
        self.repeat = int(torch_dec.repeat)
        self.conv_in = CausalConv3d(
            torch_dec.conv_in.conv.weight.detach(), torch_dec.conv_in.conv.bias.detach(), device=device, dtype=dtype
        )
        self.mid_block = MidBlock(torch_dec.mid_block, device=device, dtype=dtype)
        self.up_blocks = [UpBlock3D(ub, device=device, dtype=dtype) for ub in torch_dec.up_blocks]
        self.norm_out = RMSNorm(torch_dec.norm_out.gamma.detach(), device=device, dtype=dtype)
        self.conv_out = CausalConv3d(
            torch_dec.conv_out.conv.weight.detach(), torch_dec.conv_out.conv.bias.detach(), device=device, dtype=dtype
        )

    def __call__(self, x_bthwc):
        h = self.conv_in(x_bthwc)
        h = ttnn.add(h, ttnn.repeat_interleave(x_bthwc, self.repeat, dim=4))  # conv_in residual
        h = self.mid_block(h)
        for ub in self.up_blocks:
            h = ub(h)
        h = self.norm_out(h)
        h = silu(h)
        return self.conv_out(h)
