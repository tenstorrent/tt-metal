# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch golden model for the ttml SongUNet (edm_unet.py), 1:1 parameter map.

DDPM++ "SongUNet" in the EDM CIFAR-10 configuration (Karras et al. 2022):
model_channels=128, channel_mult=[2,2,2], num_blocks=4, attention at 16x16
(plus the middle block at 8x8), dropout=0.13, GroupNorm(eps=1e-6,
groups=min(32, C//4)), embedding dim = model_channels*4 = 512,
skip_scale = sqrt(1/2) on residual and attention adds.

Residual block (EDM UNetBlock with adaptive_scale=False, i.e. the additive
embedding path SongUNet actually uses):

    h = conv0(silu(norm0(x)))
    h = h + affine(silu(emb))          broadcast over H,W
    h = conv1(dropout(silu(norm1(h))))
    x = (skip(x) + h) * sqrt(1/2)      skip = 1x1 conv iff C changes
    if attn: x = (x + proj(SDPA(qkv(norm2(x))))) * sqrt(1/2)

DOCUMENTED DEVIATIONS from official EDM SongUNet (kept deliberately simple
for the ttml bring-up; every deviation applies identically to both the torch
golden and the ttml model, so they stay bit-comparable):
  1. Down/upsampling is a plain avgpool-2x2 / nearest-2x applied BEFORE the
     level's first block, instead of being fused into that block's conv0 and
     skip paths (EDM resamples inside UNetBlock with filter [1,1]).
  2. Per-block conditioning is affine(silu(emb)); EDM applies affine(emb)
     directly (its emb is already silu-terminated by the map layers).
  3. Embedding MLP mirrors edm.py / the DiT: feats(512) come from
     timestep_features(c_noise * CNOISE_FEAT_SCALE, 512) host-side, then
     emb = t_fc2(silu(t_fc1(feats))) + label_emb(onehot). EDM instead uses a
     128-dim Fourier feature -> silu after BOTH map layers, and adds the
     class embedding before the MLP.
  4. Attention projections are init'd normal(0, 0.02) for qkv and zeros for
     proj (EDM uses xavier-based init_attn / init_zero); conv1 and out_conv
     are zero-init like EDM's init_zero.
  5. Attention is single-head (matches DDPM++/EDM SongUNet num_heads=1).

CONV WEIGHT ROW_ORDER (the load-bearing convention):
    Every trainable 3x3 conv weight is stored FLATTENED as a matrix
        W_flat [9*C_in, C_out]
    with row index  r = (kh*3 + kw)*C_in + c_in ,  kh, kw in {0,1,2}
    scanning the kernel window top-left -> bottom-right (row-major), input
    channel fastest. I.e. flatten order = (kh, kw, c_in).
    Equivalent OIHW view:  W_flat.view(3,3,Cin,Cout).permute(3,2,0,1).
    The device im2col concatenates its 9 shifted slices in the same
    (kh, kw) order, each contributing C_in channels, so
    patches [BHW, 9*C_in] @ W_flat == conv2d(x, OIHW, padding=1).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edm import build_unet_plan

SKIP_SCALE = math.sqrt(0.5)


class Conv3x3Flat(nn.Module):
    """3x3 same-pad conv whose weight is stored flattened [9*Cin, Cout].

    ROW_ORDER = (kh, kw, c_in): see the module docstring. forward() views the
    flat weight back to OIHW and calls F.conv2d, so semantics are standard.
    """

    def __init__(self, cin: int, cout: int, zero_init: bool = False):
        super().__init__()
        self.cin, self.cout = cin, cout
        fan_in = 9 * cin
        k = 1.0 / math.sqrt(fan_in)
        w = torch.zeros(fan_in, cout) if zero_init else torch.empty(fan_in, cout).uniform_(-k, k)
        b = torch.zeros(cout) if zero_init else torch.empty(cout).uniform_(-k, k)
        self.weight = nn.Parameter(w)  # [9*Cin, Cout], ROW_ORDER (kh, kw, cin)
        self.bias = nn.Parameter(b)

    def oihw(self) -> torch.Tensor:
        return self.weight.view(3, 3, self.cin, self.cout).permute(3, 2, 0, 1)

    def forward(self, x):  # x [B, Cin, H, W]
        return F.conv2d(x, self.oihw(), self.bias, padding=1)


class Conv1x1(nn.Module):
    """1x1 conv with nn.Linear-style params [out, in] (mirrors ttml LinearLayer)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        k = 1.0 / math.sqrt(cin)
        self.weight = nn.Parameter(torch.empty(cout, cin).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(cout).uniform_(-k, k))

    def forward(self, x):  # [B, Cin, H, W]
        return torch.einsum("oc,bchw->bohw", self.weight, x) + self.bias.view(1, -1, 1, 1)


class AttentionTokens(nn.Module):
    """Single-head self-attention over flattened spatial tokens (EDM style)."""

    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.normal_(self.qkv.weight, 0.0, 0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):  # [B, C, H, W]
        b, c, h, w = x.shape
        t = x.reshape(b, c, h * w).transpose(1, 2)  # [B, T, C]
        q, k, v = self.qkv(t).chunk(3, dim=-1)
        o = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        return self.proj(o).transpose(1, 2).reshape(b, c, h, w)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c, eps=1e-6)


class UNetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, attn: bool, dropout: float):
        super().__init__()
        self.dropout_p = dropout
        self.norm0 = _gn(cin)
        self.conv0 = Conv3x3Flat(cin, cout)
        self.affine = nn.Linear(emb_dim, cout)
        self.norm1 = _gn(cout)
        self.conv1 = Conv3x3Flat(cout, cout, zero_init=True)
        self.skip = Conv1x1(cin, cout) if cin != cout else None
        if attn:
            self.norm2 = _gn(cout)
            self.attn = AttentionTokens(cout)
        else:
            self.norm2 = self.attn = None

    def forward(self, x, emb):  # x [B,Cin,H,W], emb [B,emb_dim]
        h = self.conv0(F.silu(self.norm0(x)))
        h = h + self.affine(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm1(h))
        h = self.conv1(F.dropout(h, self.dropout_p, self.training))
        x = (h + (self.skip(x) if self.skip is not None else x)) * SKIP_SCALE
        if self.attn is not None:
            x = (x + self.attn(self.norm2(x))) * SKIP_SCALE
        return x


class TorchSongUNet(nn.Module):
    """Golden CPU mirror of edm_unet.SongUNet. See module docstring for spec."""

    def __init__(
        self,
        img_resolution: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        label_dim: int = 10,  # a +1 null class is appended (CFG convention)
        model_channels: int = 128,
        channel_mult: tuple = (2, 2, 2),
        num_blocks: int = 4,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.13,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        emb_dim = model_channels * 4
        self.emb_dim = emb_dim
        self.plan_enc, self.plan_dec = build_unet_plan(
            img_resolution, model_channels, channel_mult, num_blocks, attn_resolutions
        )
        self.conv_in = Conv3x3Flat(in_channels, model_channels)
        self.t_fc1 = nn.Linear(emb_dim, emb_dim)
        self.t_fc2 = nn.Linear(emb_dim, emb_dim)
        self.label_emb = nn.Linear(label_dim + 1, emb_dim, bias=False)
        self.enc = nn.ModuleList(UNetBlock(s.cin, s.cout, emb_dim, s.attn, dropout) for s in self.plan_enc)
        self.dec = nn.ModuleList(UNetBlock(s.cin, s.cout, emb_dim, s.attn, dropout) for s in self.plan_dec)
        cfinal = self.plan_dec[-1].cout
        self.out_norm = _gn(cfinal)
        self.out_conv = Conv3x3Flat(cfinal, out_channels, zero_init=True)

    def forward(self, x, t_feats, labels_onehot, verbose: bool = False):
        """x [B,C,H,W]; t_feats [B,1,1,emb_dim]; labels_onehot [B,1,1,label_dim+1]."""

        def log(tag, t):
            if verbose:
                print(f"  {tag:<26s} {tuple(t.shape)}", flush=True)

        emb = self.t_fc2(F.silu(self.t_fc1(t_feats))) + self.label_emb(labels_onehot)
        emb = emb.reshape(emb.shape[0], self.emb_dim)  # [B, emb_dim]
        x = self.conv_in(x)
        log("conv_in", x)
        skips = [x]
        for block, spec in zip(self.enc, self.plan_enc):
            if spec.resample == "down":
                x = F.avg_pool2d(x, 2)
            x = block(x, emb)
            skips.append(x)
            log(f"enc {spec.res}x{spec.res} -> {spec.cout}" + (" attn" if spec.attn else ""), x)
        for block, spec in zip(self.dec, self.plan_dec):
            if spec.resample == "up":
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            if spec.skip_in:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
            log(f"dec {spec.res}x{spec.res} -> {spec.cout}" + (" attn" if spec.attn else ""), x)
        assert len(skips) == 0, f"decoder left {len(skips)} skips unconsumed"
        x = self.out_conv(F.silu(self.out_norm(x)))
        log("out_conv", x)
        return x


# ---------------------------------------------------------------------------
# ttml .npz checkpoint interop (mirrors cpu_sample.load_ttml_npz conventions)
# ---------------------------------------------------------------------------
#
# ttml names come from the module tree: "SongUNet/enc/3/conv0/weight" etc.
#   ttml LinearLayer  weight [1,1,out,in], bias [1,1,1,out]
#   ttml Conv3x3      weight [1,1,9*Cin,Cout] (ROW_ORDER), bias [1,1,1,Cout]
#   ttml GroupNorm    gamma/beta [1,1,1,C]
# Torch counterparts use the SAME attribute names, so translation is
# mechanical: '.'->'/' with root 'SongUNet', GN weight/bias <-> gamma/beta.


def ttml_name_map(model: TorchSongUNet) -> dict:
    """torch state_dict key -> (ttml npz key, torch shape)."""
    mapping = {}
    for mod_name, module in model.named_modules():
        prefix = "SongUNet" + ("/" + mod_name.replace(".", "/") if mod_name else "")
        if isinstance(module, nn.GroupNorm):
            mapping[f"{mod_name}.weight"] = f"{prefix}/gamma"
            mapping[f"{mod_name}.bias"] = f"{prefix}/beta"
        elif isinstance(module, (nn.Linear, Conv1x1, Conv3x3Flat)):
            mapping[f"{mod_name}.weight"] = f"{prefix}/weight"
            if getattr(module, "bias", None) is not None:
                mapping[f"{mod_name}.bias"] = f"{prefix}/bias"
    return mapping


def load_ttml_npz(model: TorchSongUNet, path: str) -> TorchSongUNet:
    """Load a tt-train SongUNet .npz checkpoint into the torch golden model."""
    z = np.load(path)
    sd = {}
    for torch_key, ttml_key in ttml_name_map(model).items():
        ref = model.state_dict()[torch_key]
        sd[torch_key] = torch.from_numpy(z[ttml_key].reshape(ref.shape).copy())
    model.load_state_dict(sd, strict=True)
    return model


def export_torch_npz(model: TorchSongUNet, path: str) -> None:
    """Save torch params under ttml names/shapes (device parity tests load this)."""
    out = {}
    sd = model.state_dict()
    for torch_key, ttml_key in ttml_name_map(model).items():
        t = sd[torch_key].detach().float().numpy()
        if t.ndim == 1:  # biases / GN affine -> [1,1,1,C]
            t = t.reshape(1, 1, 1, -1)
        elif t.ndim == 2:  # linear [o,i] / conv flat [9ci,co] -> [1,1,a,b]
            t = t.reshape(1, 1, *t.shape)
        out[ttml_key] = t
    np.savez(path, **out)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
