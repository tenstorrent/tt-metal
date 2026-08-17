# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SongUNet (DDPM++, EDM CIFAR-10 config) on ttml — mirrors reference_unet.py 1:1.

Every trainable tensor has the same name and (up to the [1,1,...] rank-4
padding) the same shape as its torch counterpart; see
reference_unet.ttml_name_map. Architecture spec, documented deviations from
official EDM, and the conv-weight ROW_ORDER all live in reference_unet.py's
module docstring — this file only mirrors it.

Activations are [B, 1, H*W, C] channels-last tokens in TILE layout, with the
current H, W tracked as python ints (see edm_ops.py for the layout policy).
1x1 convs and the embedding/attention projections are plain LinearLayers;
3x3 convs are Conv3x3Im2col with the flattened [1,1,9*Cin,Cout] weight.

Host-side input prep (see edm.py):
    net_in, feats, onehot, target = make_edm_batch_image(...)
    x_tokens      = nchw_to_nhwc_tokens(net_in)   # [B,1,H*W,3]
    target_tokens = nchw_to_nhwc_tokens(target)
    pred = model(from_numpy(x_tokens), from_numpy(feats), from_numpy(onehot))
    loss = ttml.ops.loss.mse_loss(pred, from_numpy(target_tokens))
"""

from __future__ import annotations

import math

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter, RunMode
from ttml.modules.module_base import ModuleList

from edm import build_unet_plan
from edm_ops import AvgPool2x2, ConcatChannels, Conv3x3Im2col, GroupNormMoreh, Scale, UpsampleNearest2

SKIP_SCALE = math.sqrt(0.5)


class Conv3x3(AbstractModuleBase):
    """Trainable 3x3 conv; weight stored flat [1,1,9*Cin,Cout] (ROW_ORDER =
    (kh, kw, c_in), see reference_unet.py). Init is kaiming-equivalent on the
    unflattened shape: uniform(+-1/sqrt(9*Cin)) — element-iid, so flatten
    order does not affect the distribution."""

    def __init__(self, cin: int, cout: int, zero_init: bool = False) -> None:
        super().__init__()
        self.cin, self.cout = cin, cout
        k = 1.0 / math.sqrt(9 * cin)
        w_init = ttml.init.zeros() if zero_init else ttml.init.uniform(-k, k)
        b_init = ttml.init.zeros() if zero_init else ttml.init.uniform(-k, k)
        self.weight = Parameter(w_init((1, 1, 9 * cin, cout)))
        self.bias = Parameter(b_init((1, 1, 1, cout)))

    def forward(self, x, h: int, w: int):
        return Conv3x3Im2col.apply(x, self.weight.tensor, self.bias.tensor, h, w)


class GroupNorm(AbstractModuleBase):
    """GroupNorm(min(32, C//4) groups, eps=1e-6) via the moreh kernel pair."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_groups = min(32, channels // 4)
        self.gamma = Parameter(ttml.init.ones()((1, 1, 1, channels)))
        self.beta = Parameter(ttml.init.zeros()((1, 1, 1, channels)))

    def forward(self, x, h: int, w: int):
        return GroupNormMoreh.apply(x, self.gamma.tensor, self.beta.tensor, self.num_groups, h, w)


class UNetAttention(AbstractModuleBase):
    """Single-head self-attention over the [B,1,HW,C] tokens (EDM style):
    qkv linear -> composite SDPA (mask=None, true non-causal) -> zero-init proj."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = LinearLayer(dim, 3 * dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())
        self.proj = LinearLayer(dim, dim, True, weight_init=ttml.init.zeros(), bias_init=ttml.init.zeros())

    def forward(self, x):
        q, k, v = ttml.ops.multi_head_utils.heads_creation(self.qkv(x), 1)
        out = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, None)
        return self.proj(ttml.ops.multi_head_utils.heads_fusion(out))


class UNetBlock(AbstractModuleBase):
    """DDPM++ residual block, additive embedding (reference_unet.UNetBlock)."""

    def __init__(self, cin: int, cout: int, emb_dim: int, attn: bool, dropout: float) -> None:
        super().__init__()
        self.dropout_p = dropout
        self.norm0 = GroupNorm(cin)
        self.conv0 = Conv3x3(cin, cout)
        self.affine = LinearLayer(emb_dim, cout, True)
        self.norm1 = GroupNorm(cout)
        self.conv1 = Conv3x3(cout, cout, zero_init=True)
        self.skip = LinearLayer(cin, cout, True) if cin != cout else None
        if attn:
            self.norm2 = GroupNorm(cout)
            self.attn = UNetAttention(cout)
        else:
            self.norm2 = self.attn = None

    def forward(self, x, emb, h: int, w: int):
        add, silu = ttml.ops.binary.add, ttml.ops.unary.silu
        y = self.conv0(silu(self.norm0(x, h, w)), h, w)
        y = add(y, self.affine(silu(emb)))  # [B,1,HW,C] + [B,1,1,C] row broadcast
        y = silu(self.norm1(y, h, w))
        if self.dropout_p > 0.0 and self.get_run_mode() == RunMode.TRAIN:
            y = ttml.ops.dropout.dropout(y, self.dropout_p)
        y = self.conv1(y, h, w)
        sk = self.skip(x) if self.skip is not None else x
        x = Scale.apply(add(sk, y), SKIP_SCALE)
        if self.attn is not None:
            x = Scale.apply(add(x, self.attn(self.norm2(x, h, w))), SKIP_SCALE)
        return x


class SongUNet(AbstractModuleBase):
    """EDM CIFAR-10 DDPM++ SongUNet. forward(x_tokens, t_feats, labels_onehot):

        x_tokens      [B, 1, H*W, in_channels]   channels-last tokens
        t_feats       [B, 1, 1, emb_dim]         host timestep_features(cnoise*scale)
        labels_onehot [B, 1, 1, label_dim+1]
        returns       [B, 1, H*W, out_channels]
    """

    def __init__(
        self,
        img_resolution: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        label_dim: int = 10,
        model_channels: int = 128,
        channel_mult: tuple = (2, 2, 2),
        num_blocks: int = 4,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.13,
    ) -> None:
        super().__init__()
        self.img_resolution = img_resolution
        emb_dim = model_channels * 4
        self.emb_dim = emb_dim
        self.plan_enc, self.plan_dec = build_unet_plan(
            img_resolution, model_channels, channel_mult, num_blocks, attn_resolutions
        )
        self.conv_in = Conv3x3(in_channels, model_channels)
        self.t_fc1 = LinearLayer(emb_dim, emb_dim, True)
        self.t_fc2 = LinearLayer(emb_dim, emb_dim, True)
        self.label_emb = LinearLayer(label_dim + 1, emb_dim, False)
        self.enc = ModuleList([UNetBlock(s.cin, s.cout, emb_dim, s.attn, dropout) for s in self.plan_enc])
        self.dec = ModuleList([UNetBlock(s.cin, s.cout, emb_dim, s.attn, dropout) for s in self.plan_dec])
        cfinal = self.plan_dec[-1].cout
        self.out_norm = GroupNorm(cfinal)
        self.out_conv = Conv3x3(cfinal, out_channels, zero_init=True)

    def forward(self, x, t_feats, labels_onehot):
        add, silu = ttml.ops.binary.add, ttml.ops.unary.silu
        emb = add(self.t_fc2(silu(self.t_fc1(t_feats))), self.label_emb(labels_onehot))
        h = w = self.img_resolution
        x = self.conv_in(x, h, w)
        skips = [x]
        for block, spec in zip(self.enc, self.plan_enc):
            if spec.resample == "down":
                x = AvgPool2x2.apply(x, h, w)
                h, w = h // 2, w // 2
            x = block(x, emb, h, w)
            skips.append(x)
        for block, spec in zip(self.dec, self.plan_dec):
            if spec.resample == "up":
                x = UpsampleNearest2.apply(x, h, w)
                h, w = h * 2, w * 2
            if spec.skip_in:
                x = ConcatChannels.apply(x, skips.pop())
            x = block(x, emb, h, w)
        assert not skips, f"decoder left {len(skips)} skips unconsumed"
        return self.out_conv(silu(self.out_norm(x, h, w)), h, w)


def song_unet_cifar(label_dim: int = 10) -> SongUNet:
    """The exact EDM CIFAR-10 DDPM++ config (55-62M params)."""
    return SongUNet(label_dim=label_dim)


def song_unet_tiny(img_resolution: int = 32, label_dim: int = 10) -> SongUNet:
    """Bring-up config: one block per level, narrow channels, no dropout."""
    return SongUNet(
        img_resolution=img_resolution,
        label_dim=label_dim,
        model_channels=32,
        num_blocks=1,
        dropout=0.0,
    )
